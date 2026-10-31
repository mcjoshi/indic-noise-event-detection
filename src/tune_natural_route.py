"""Tune the decode config for the natural route only, with the refiner in the loop.

The routed system sends natural clips to one model and synthetic clips to another, so the
natural route's decode parameters should be fitted on natural clips alone.  Fitting them
on the 62/38 mixture is what produced the compromise this whole routing design exists to
undo.

The refiner is applied inside the objective rather than afterwards, because it shifts
boundaries and therefore changes which threshold and smoothing are optimal - tuning
without it and then bolting it on would optimise the wrong operating point.
"""

from __future__ import annotations

import argparse
import os
import json

import numpy as np
import torch

from boundary_refiner import refine_spans
import multiprocessing as mp

_CTX: dict = {}


W_F1, W_DICE = 1.0, 1.0


def _eval_cfg(arg):
    """Pool worker: score one decode config on a clip subset (context inherited by fork)."""
    cfg, subset_ids = arg
    torch.set_num_threads(1)
    probs, bnd, durations, refs, refiner = (_CTX[k] for k in ("probs", "bnd", "durations", "refs", "refiner"))
    preds = {}
    for cid in subset_ids:
        spans = decode_one(probs[cid], durations[cid], cfg, bnd.get(cid))
        if refiner is not None:
            on, off = bnd[cid]
            spans = refine_spans(spans, probs[cid], on, off, refiner, "cpu", durations[cid])
        preds[cid] = spans
    _s = score({k: refs[k] for k in subset_ids}, preds, {k: durations[k] for k in subset_ids})
    return W_F1 * _s.event_f1 + W_DICE * _s.segment_dice
from ensemble import load_cache
from scorer import score
from tune_postproc import DEFAULT_GRID, BASE_CFG, decode_one


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--refiner", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--w-f1", type=float, default=1.0, help="objective weight on event F1 (board natural weight 0.4376)")
    ap.add_argument("--w-dice", type=float, default=1.0, help="objective weight on segment Dice (board natural weight 0.6109)")
    args = ap.parse_args()
    global W_F1, W_DICE
    W_F1, W_DICE = args.w_f1, args.w_dice

    probs, durations, records, bnd = load_cache(args.cache)
    nat = [r for r in records if not r.get("synthetic")]
    ids = {r["clip_id"] for r in nat}
    refs = {r["clip_id"]: [(e["onset"], e["offset"]) for e in r["events"]] for r in nat}
    print(f"tuning on {len(ids)} natural clips")

    device = "cpu"
    refiner = None
    if args.refiner:
        from boundary_refiner import load_refiner
        refiner = load_refiner(args.refiner, device)
        refiner.eval()
        print(f"refiner in the loop: {args.refiner}")

    def evaluate(cfg):
        preds = {}
        for cid in ids:
            spans = decode_one(probs[cid], durations[cid], cfg, bnd.get(cid))
            if refiner is not None:
                on, off = bnd[cid]
                spans = refine_spans(spans, probs[cid], on, off, refiner, device,
                                     durations[cid])
            preds[cid] = spans
        return score(refs, preds, {k: durations[k] for k in ids})

    speakers = sorted({r["speaker"] for r in nat})
    rng = np.random.default_rng(0)
    rng.shuffle(speakers)
    fold = {s: i % 2 for i, s in enumerate(speakers)}
    by_fold = {f: {r["clip_id"] for r in nat if fold[r["speaker"]] == f} for f in (0, 1)}

    _CTX.update(probs=probs, bnd=bnd, durations=durations, refs=refs, refiner=refiner)
    pool = mp.get_context("fork").Pool(min(12, os.cpu_count() or 1))

    def search(subset_ids, verbose=False):
        cfg = dict(BASE_CFG)

        def ev(c):
            preds = {}
            for cid in subset_ids:
                spans = decode_one(probs[cid], durations[cid], c, bnd.get(cid))
                if refiner is not None:
                    on, off = bnd[cid]
                    spans = refine_spans(spans, probs[cid], on, off, refiner, device,
                                         durations[cid])
                preds[cid] = spans
            _o = score({k: refs[k] for k in subset_ids}, preds,
                       {k: durations[k] for k in subset_ids})
            return W_F1 * _o.event_f1 + W_DICE * _o.segment_dice

        best = ev(cfg)
        order = ["threshold", "median_k", "offset_threshold", "min_dur",
                 "merge_gap", "snap_sec", "onset_bias", "offset_bias"]
        for _ in range(args.rounds):
            improved = False
            for param in order:
                cur, val = best, cfg[param]
                trials = []
                for v in DEFAULT_GRID[param]:
                    if v == cfg[param]:
                        continue
                    trial = dict(cfg, **{param: v})
                    if (trial["offset_threshold"] is not None
                            and trial["offset_threshold"] >= trial["threshold"]):
                        continue
                    trials.append((v, trial))
                scores = pool.map(_eval_cfg, [(t, subset_ids) for _, t in trials]) if trials else []
                for (v, _), s in zip(trials, scores):
                    if s > cur:
                        cur, val = s, v
                if cur > best + 1e-6:
                    best, cfg[param], improved = cur, val, True
                    if verbose:
                        print(f"    {param:<18} -> {val!r:8}  {best:.4f}")
            if not improved:
                break
        return cfg, best

    oof = []
    for f in (0, 1):
        cfg_f, fit = search(by_fold[f])
        preds = {}
        for cid in by_fold[1 - f]:
            spans = decode_one(probs[cid], durations[cid], cfg_f, bnd.get(cid))
            if refiner is not None:
                on, off = bnd[cid]
                spans = refine_spans(spans, probs[cid], on, off, refiner, device,
                                     durations[cid])
            preds[cid] = spans
        s = score({k: refs[k] for k in by_fold[1 - f]}, preds,
                  {k: durations[k] for k in by_fold[1 - f]})
        oof.append(s.combined)
        print(f"  fold {f}: fit {fit:.4f} -> out-of-fold {s.combined:.4f}")

    print("\nfinal fit on all natural clips")
    cfg, _ = search(ids, verbose=True)
    s = evaluate(cfg)
    print(f"\nnatural-route config: {json.dumps(cfg)}")
    print(f"  natural Combined {s.combined:.4f}  F1 {s.event_f1:.4f}  Dice {s.segment_dice:.4f}")
    print(f"  out-of-fold estimate {np.mean(oof):.4f}")
    with open(args.out, "w") as fh:
        json.dump({"config": cfg, "natural_scores": s.as_dict(),
                   "oof_estimate": round(float(np.mean(oof)), 5)}, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
