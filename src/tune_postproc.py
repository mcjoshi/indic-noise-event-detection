"""Search decoding parameters against the local scorer.

Post-processing is where this metric is actually won or lost, so the search is staged
rather than a single product grid: frame probabilities are computed once and cached,
then parameters are optimised in rounds of coordinate descent.  A full product over
seven parameters would be ~10^6 configurations; coordinate descent over cached
probabilities reaches the same neighbourhood in a few hundred.

The search maximises Combined (Event-F1 + Dice) directly, because the two terms pull in
opposite directions: filling a gap raises Dice while merging two reference events into
one prediction costs two Event-F1 matches.  Optimising either term alone picks the wrong
operating point.
"""

from __future__ import annotations

import argparse
import os
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import VaaniFrameDataset, collate
from model import decode, load_checkpoint, snap_to_boundaries, tta_forward
from scorer import score


def cache_probs(ckpt_path, val_jsonl, audio_root, device, batch_size=None, workers=8, use_ema=True, residual_root=None):
    # VAANI_BATCH lets a run fit beside other jobs on a shared GPU
    batch_size = batch_size or int(os.environ.get("VAANI_BATCH", "32"))
    model, _ = load_checkpoint(ckpt_path, device, use_ema=use_ema)

    ds = VaaniFrameDataset(val_jsonl, audio_root, max_sec=None, train=False, residual_root=residual_root)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                    collate_fn=collate, pin_memory=True)

    probs, durations, bnd = {}, {}, {}
    with torch.no_grad():
        for batch in dl:
            wav = batch["wav"].to(device, non_blocking=True)
            n_fr = int(batch["frame_mask"].shape[1])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
                res_t = batch["res"].to(device, non_blocking=True) if batch.get("res") is not None else None
                out = tta_forward(model, wav, n_fr, res_t) if os.environ.get("TTA_SHIFT") == "1" else model(wav, n_fr, res_t)
            logits, log_on, log_off = out[0], out[2], out[3]
            p = torch.sigmoid(logits.float()).cpu().numpy()
            pon = torch.sigmoid(log_on.float()).cpu().numpy()
            poff = torch.sigmoid(log_off.float()).cpu().numpy()
            lengths = batch["frame_mask"].sum(dim=1).long().numpy()
            for j, cid in enumerate(batch["clip_ids"]):
                k = lengths[j]
                probs[cid] = p[j, :k].astype(np.float32)
                bnd[cid] = (pon[j, :k].astype(np.float32), poff[j, :k].astype(np.float32))
                durations[cid] = batch["durations"][j]
    return probs, durations, ds.records, bnd


def decode_one(p, dur, cfg, bnd_pair):
    """Decode a clip, optionally snapping edges onto the boundary-head peaks.

    `snap_sec` is searched like any other decode parameter (0 disables it) so the
    boundary heads have to earn their place on the validation score rather than being
    assumed helpful.
    """
    cfg = dict(cfg)
    snap_sec = cfg.pop("snap_sec", 0.0)
    spans = decode(p, duration=dur, **cfg)
    if snap_sec and bnd_pair is not None:
        spans = snap_to_boundaries(spans, bnd_pair[0], bnd_pair[1], search_sec=snap_sec)
    return spans


def evaluate_config(probs, durations, refs, cfg, bnd=None):
    preds = {
        cid: decode_one(p, durations[cid], cfg, (bnd or {}).get(cid))
        for cid, p in probs.items()
    }
    return score(refs, preds, durations)


DEFAULT_GRID = {
    # research/03 argued for 30-150 ms median windows and short merge gaps, on the
    # reasoning that merging two references into one prediction costs two Event-F1
    # matches.  The first search pinned median_k, merge_gap and min_dur at the top of
    # those ranges, so the ranges are extended here and the question is settled by the
    # holdout rather than by the prior.
    "median_k": [1, 3, 5, 7, 9, 11, 15, 19, 23, 27, 31, 41],
    "threshold": [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
    "offset_threshold": [None, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55],
    "merge_gap": [0.0, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.2, 0.26, 0.32, 0.4, 0.5],
    "min_dur": [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.26, 0.32, 0.4],
    "onset_bias": [-0.1, -0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06],
    "offset_bias": [-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08, 0.1],
    "snap_sec": [0.0, 0.05, 0.08, 0.10, 0.13, 0.16, 0.20],
}

BASE_CFG = {
    "threshold": 0.5, "offset_threshold": None, "median_k": 5,
    "merge_gap": 0.0, "min_dur": 0.05, "onset_bias": 0.0, "offset_bias": 0.0,
    "snap_sec": 0.0,
}


def coordinate_search(probs, durations, refs, rounds=3, grid=None, verbose=True, bnd=None):
    grid = grid or DEFAULT_GRID
    cfg = dict(BASE_CFG)
    best_s = evaluate_config(probs, durations, refs, cfg, bnd)
    best = best_s.combined
    if verbose:
        print(f"start: Combined {best:.4f} (F1 {best_s.event_f1:.4f} Dice {best_s.segment_dice:.4f}) {cfg}")

    order = ["threshold", "median_k", "offset_threshold", "min_dur",
             "merge_gap", "snap_sec", "onset_bias", "offset_bias"]
    for rnd in range(rounds):
        improved = False
        for param in order:
            trial_best, trial_val = best, cfg[param]
            for value in grid[param]:
                if value == cfg[param]:
                    continue
                trial = dict(cfg, **{param: value})
                # hysteresis must genuinely be a lower offset threshold
                if trial["offset_threshold"] is not None and trial["offset_threshold"] >= trial["threshold"]:
                    continue
                s = evaluate_config(probs, durations, refs, trial, bnd)
                if s.combined > trial_best:
                    trial_best, trial_val = s.combined, value
            if trial_best > best + 1e-6:
                best, cfg[param], improved = trial_best, trial_val, True
                if verbose:
                    print(f"  round {rnd} {param:18s} -> {trial_val!r:8}  Combined {best:.4f}")
        if not improved:
            if verbose:
                print(f"  round {rnd}: converged")
            break
    return cfg, best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None, help="pickle of cached val probabilities")
    ap.add_argument("--residual-root", default=None, help="external noise view for two-channel models (default: the model's own U-Net residual)")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--no-ema", action="store_true")
    args = ap.parse_args()

    device = os.environ.get("VAANI_DEVICE", "cuda")  # "cpu" when every GPU is taken
    frac = float(os.environ.get("VAANI_MEM_FRAC", "0"))
    if frac > 0 and torch.cuda.is_available():
        # fail fast with an OOM traceback instead of deadlocking a shared card
        torch.cuda.set_per_process_memory_fraction(frac)
    # the cache is keyed by checkpoint identity: silently reusing another model's
    # probabilities would tune that model's decode config and ship it on these weights
    ckpt_key = f"{Path(args.ckpt).resolve()}::{Path(args.ckpt).stat().st_mtime_ns}::ema={not args.no_ema}::tta={os.environ.get('TTA_SHIFT', '0')}::res={args.residual_root or ''}"
    cached = None
    if args.cache and Path(args.cache).exists():
        with open(args.cache, "rb") as fh:
            blob = pickle.load(fh)
        if isinstance(blob, dict) and blob.get("key") == ckpt_key and blob.get("bnd"):
            cached = (blob["probs"], blob["durations"], blob["records"], blob["bnd"])
        elif isinstance(blob, dict) and blob.get("key") == ckpt_key:
            # a cache without boundary probabilities would make snap_sec a silent no-op,
            # so the search would "reject" snapping for a reason that is not real
            print(f"cache {args.cache} predates boundary caching - recomputing")
        else:
            print(f"cache {args.cache} was built from a different checkpoint - recomputing")

    if cached is not None:
        probs, durations, records, bnd = cached
        print(f"loaded cached probabilities for {len(probs)} clips")
    else:
        probs, durations, records, bnd = cache_probs(
            args.ckpt, args.val, args.audio_root, device, use_ema=not args.no_ema, residual_root=args.residual_root
        )
        if args.cache:
            with open(args.cache, "wb") as fh:
                pickle.dump({"key": ckpt_key, "probs": probs, "durations": durations,
                             "records": records, "bnd": bnd}, fh)
        print(f"computed probabilities for {len(probs)} clips")

    refs = {r["clip_id"]: [(e["onset"], e["offset"]) for e in r["events"]] for r in records}

    # Seven decode parameters are selected on the same 2,005 clips that chose the
    # checkpoint, so a single fit would report an optimistic number.  Two-fold
    # cross-validation over speaker-disjoint halves gives the honest out-of-fold
    # estimate; the shipped config is then refitted on all of val, which is the most
    # data available and therefore the most stable fit.
    speakers = sorted({r["speaker"] for r in records})
    rng = np.random.default_rng(0)
    rng.shuffle(speakers)
    fold_of = {s: (i % 2) for i, s in enumerate(speakers)}
    ids_of_fold = {
        f: {r["clip_id"] for r in records if fold_of[r["speaker"]] == f} for f in (0, 1)
    }

    def subset(ids):
        return (
            {k: v for k, v in probs.items() if k in ids},
            {k: v for k, v in durations.items() if k in ids},
            {k: v for k, v in refs.items() if k in ids},
            {k: v for k, v in (bnd or {}).items() if k in ids},
        )

    oof = []
    for f in (0, 1):
        fit_p, fit_d, fit_r, fit_b = subset(ids_of_fold[f])
        ev_p, ev_d, ev_r, ev_b = subset(ids_of_fold[1 - f])
        print(f"--- fold {f}: fit on {len(fit_p)} clips, score on {len(ev_p)} ---")
        cfg_f, fit_best = coordinate_search(fit_p, fit_d, fit_r, rounds=args.rounds,
                                            verbose=False, bnd=fit_b)
        s_out = evaluate_config(ev_p, ev_d, ev_r, cfg_f, ev_b)
        oof.append(s_out.combined)
        print(f"    fit {fit_best:.4f} -> out-of-fold {s_out.combined:.4f} "
              f"(gap {fit_best - s_out.combined:+.4f})  {json.dumps(cfg_f)}")

    honest = float(np.mean(oof))

    print("\n--- final fit on all of val ---")
    cfg, _ = coordinate_search(probs, durations, refs, rounds=args.rounds, bnd=bnd)
    s_full = evaluate_config(probs, durations, refs, cfg, bnd)

    print(f"\nshipped config: {json.dumps(cfg)}")
    print(f"  full-val (optimistic, config fitted here): {s_full.combined:.4f} "
          f"(F1 {s_full.event_f1:.4f} Dice {s_full.segment_dice:.4f})")
    print(f"  2-fold out-of-fold estimate:               {honest:.4f}  <- expect this on test")
    print(f"  selection optimism:                        {s_full.combined - honest:+.4f}")

    with open(args.out, "w") as fh:
        json.dump({
            "config": cfg,
            "scores_full_val": s_full.as_dict(),
            "oof_per_fold": [round(x, 5) for x in oof],
            "oof_estimate": round(honest, 5),
            "selection_optimism": round(s_full.combined - honest, 5),
        }, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
