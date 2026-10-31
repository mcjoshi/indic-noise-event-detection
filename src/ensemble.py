"""Evaluate an ensemble of cached frame probabilities.

v1 and v3 both emit probabilities on the same 10 ms grid, so their outputs align
per clip and can simply be averaged - no retraining, one extra inference pass at
submission time.  The decode config is re-tuned on the ensemble because averaging
changes the probability calibration, so v3's threshold is not the right one for it.

Reports the 2-fold out-of-fold estimate, the same protocol used for the single models,
so the numbers are directly comparable.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from tune_postproc import coordinate_search, evaluate_config


def load_cache(path: str):
    with open(path, "rb") as fh:
        blob = pickle.load(fh)
    if isinstance(blob, dict):
        return blob["probs"], blob["durations"], blob["records"], blob.get("bnd")
    probs, durations, records = blob
    return probs, durations, records, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--caches", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--natural-only", action="store_true",
                    help="fit on natural clips only - the routed system uses the\n"
                         "ensemble for the natural route alone")
    args = ap.parse_args()

    caches = [load_cache(c) for c in args.caches]
    probs0, durations, records, _ = caches[0]
    weights = args.weights or [1.0] * len(caches)
    assert len(weights) == len(caches)
    w = np.array(weights, dtype=np.float64)
    w = w / w.sum()
    print("members: " + ", ".join(
        f"{Path(c).stem} (w={wi:.2f})" for c, wi in zip(args.caches, w)
    ))

    common = set(probs0)
    for p, _, _, _ in caches[1:]:
        common &= set(p)
    print(f"{len(common)} clips common to all members")

    ens = {}
    for cid in common:
        stack = [c[0][cid].astype(np.float64) for c in caches]
        n = min(len(s) for s in stack)
        ens[cid] = np.average([s[:n] for s in stack], axis=0, weights=w).astype(np.float32)

    if args.natural_only:
        records = [r for r in records if not r.get("synthetic")]
        common &= {r["clip_id"] for r in records}
        ens = {k: v for k, v in ens.items() if k in common}
        print(f"natural-only: {len(common)} clips")
    durations = {k: v for k, v in durations.items() if k in common}
    records = [r for r in records if r["clip_id"] in common]
    refs = {r["clip_id"]: [(e["onset"], e["offset"]) for e in r["events"]] for r in records}

    # individual members, for a like-for-like baseline under this same protocol
    for c, path in zip(caches, args.caches):
        sub = {k: c[0][k] for k in common}
        cfg_i, _ = coordinate_search(sub, durations, refs, rounds=args.rounds, verbose=False)
        s_i = evaluate_config(sub, durations, refs, cfg_i)
        print(f"  {Path(path).stem:18s} full-val {s_i.combined:.4f} "
              f"(F1 {s_i.event_f1:.4f} Dice {s_i.segment_dice:.4f})")

    speakers = sorted({r["speaker"] for r in records})
    rng = np.random.default_rng(0)
    rng.shuffle(speakers)
    fold_of = {s: i % 2 for i, s in enumerate(speakers)}
    ids = {f: {r["clip_id"] for r in records if fold_of[r["speaker"]] == f} for f in (0, 1)}

    def subset(keep):
        return ({k: v for k, v in ens.items() if k in keep},
                {k: v for k, v in durations.items() if k in keep},
                {k: v for k, v in refs.items() if k in keep})

    oof = []
    for f in (0, 1):
        fp, fd, fr = subset(ids[f])
        ep, ed, er = subset(ids[1 - f])
        cfg_f, fit = coordinate_search(fp, fd, fr, rounds=args.rounds, verbose=False)
        s_out = evaluate_config(ep, ed, er, cfg_f)
        oof.append(s_out.combined)
        print(f"  ensemble fold {f}: fit {fit:.4f} -> out-of-fold {s_out.combined:.4f}")

    print("\n--- ensemble, final fit on all of val ---")
    cfg, _ = coordinate_search(ens, durations, refs, rounds=args.rounds)
    s = evaluate_config(ens, durations, refs, cfg)
    honest = float(np.mean(oof))
    print(f"\nensemble config: {json.dumps(cfg)}")
    print(f"  full-val (optimistic): {s.combined:.4f} "
          f"(F1 {s.event_f1:.4f} Dice {s.segment_dice:.4f})")
    print(f"  2-fold out-of-fold:    {honest:.4f}  <- expect this on test")

    with open(args.out, "w") as fh:
        json.dump({"config": cfg, "members": args.caches, "weights": list(w),
                   "scores_full_val": s.as_dict(), "oof_estimate": round(honest, 5)}, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
