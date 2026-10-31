"""Distillation labels from the fused ensemble: Silver boundaries moved onto fused events where
justified (IoU >= min_iou, movement capped - the annotator's "an event is here" is kept, only its
placement changes); Bronze clips get the fused events as pseudo timestamps (tier bronze_pseudo).
Gold is untouched."""
from __future__ import annotations
from paths import DATA, PRETRAINED, WORK  # noqa: F401

import argparse
import json

import numpy as np

from wbf_ensemble import fuse


def load(path):
    out = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            out[r["clip_id"]] = [(float(e["onset"]), float(e["offset"])) for e in r["events"]]
    return out


def move_boundaries(original, cand, min_iou, max_shift):
    out, moved = [], 0
    for ev in original:
        o_on, o_off = float(ev["onset"]), float(ev["offset"])
        best, best_iou = None, 0.0
        for c_on, c_off in cand:
            inter = max(0.0, min(o_off, c_off) - max(o_on, c_on))
            union = max(o_off, c_off) - min(o_on, c_on)
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best, best_iou = (c_on, c_off), iou
        if best is not None and best_iou >= min_iou:
            n_on = float(np.clip(best[0], o_on - max_shift, o_on + max_shift))
            n_off = float(np.clip(best[1], o_off - max_shift, o_off + max_shift))
            if n_off - n_on >= 0.05:
                out.append(dict(ev, onset=round(n_on, 4), offset=round(n_off, 4)))
                moved += 1
                continue
        out.append(ev)
    return out, moved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teachers", nargs="+", required=True)
    ap.add_argument("--manifest", default=f"{DATA}/splits/train_v7.jsonl")
    ap.add_argument("--extra-manifests", nargs="*", default=[], help="e.g. the parent-corpus pool")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--silver-votes", type=int, default=2)
    ap.add_argument("--bronze-votes", type=int, default=3)
    ap.add_argument("--min-iou", type=float, default=0.3)
    ap.add_argument("--max-shift", type=float, default=0.5)
    args = ap.parse_args()
    teachers = [load(p) for p in args.teachers]
    records = [json.loads(l) for l in open(args.manifest) if l.strip()]
    for m in args.extra_manifests:
        records += [json.loads(l) for l in open(m) if l.strip()]
    n_silver = n_moved = n_ev = n_bronze = n_bronze_ev = n_parent = n_parent_ev = 0
    shifts = []
    with open(args.out, "w") as fh:
        for r in records:
            cid = r["clip_id"]
            if r["tier"] == "silver" and r["events"] and all(cid in t for t in teachers):
                cand = fuse([t[cid] for t in teachers], args.iou, args.silver_votes)
                new, moved = move_boundaries(r["events"], cand, args.min_iou, args.max_shift)
                for a, b in zip(r["events"], new):
                    shifts += [abs(float(a["onset"]) - float(b["onset"])), abs(float(a["offset"]) - float(b["offset"]))]
                n_silver += 1; n_moved += moved; n_ev += len(new)
                r = dict(r, events=new, tier_refined=True)
            elif r["tier"] == "parent" and all(cid in t for t in teachers):
                # unlabeled in-domain clip: keep it only when the ensemble agrees an event exists
                cand = fuse([t[cid] for t in teachers], args.iou, args.bronze_votes)
                if not cand:
                    continue
                r = dict(r, events=[{"onset": a, "offset": b} for a, b in cand], tier="parent_pseudo")
                n_parent += 1; n_parent_ev += len(cand)
            elif r["tier"] in ("bronze", "bronze_pseudo") and all(cid in t for t in teachers):
                cand = fuse([t[cid] for t in teachers], args.iou, args.bronze_votes)
                if cand:
                    r = dict(r, events=[{"onset": a, "offset": b} for a, b in cand], tier="bronze_pseudo")
                    n_bronze += 1; n_bronze_ev += len(cand)
                else:
                    r = dict(r, events=[], tier="bronze")
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    shifts = np.array(shifts) if shifts else np.zeros(1)
    print(f"silver: {n_silver} clips, moved {n_moved}/{n_ev} events; movement mean {shifts.mean()*1000:.0f} ms "
          f"median {np.median(shifts)*1000:.0f} ms p95 {np.percentile(shifts,95)*1000:.0f} ms")
    print(f"bronze: {n_bronze} clips carry {n_bronze_ev} fused events ({n_bronze_ev/max(n_bronze,1):.2f}/clip)")
    print(f"parent: {n_parent} clips kept with {n_parent_ev} fused events ({n_parent_ev/max(n_parent,1):.2f}/clip)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
