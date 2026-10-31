"""Fuse the teacher routes over the pseudo-label pools and write the stage-2 training manifest.

Teacher = v137, v161, v273 (threshold + proposal routes), fused like the board set (IoU 0.6, median edges, fallback).
Rules (the annotator's "an event is here" is kept; only its placement may change):
  Silver rows        keep every annotator event; move its edges onto the 3-vote teacher span when IoU >= 0.3 (shift capped
                     at 0.5 s); add a teacher event only where >= 5 of the 6 routes agree and it overlaps no annotator event;
                     tier stays silver (soft targets).
  bronze_pseudo and  these carry the old teacher's labels, not human events: replace them with the new 3-vote fusion,
  parent_pseudo rows tier unchanged.
  Gold and bronze    copied unchanged.
usage: python3 pl_fuse.py [--votes 3] [--add-votes 5] [--out data/splits/train_v29_pl2.jsonl]
"""
from __future__ import annotations
from paths import DATA, PRETRAINED, WORK  # noqa: F401

import argparse
import collections
import json

from pseudo_label_fused import move_boundaries
from wbf_ensemble import fuse

W = f"{WORK}/pl"
TEACHERS = ["v137_n3_longunf", "v161_v145_dice3_s2", "v273_hubert_unf"]
SRC = f"{DATA}/splits/train_v29.jsonl"


def load(path):
    out = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            out[r["clip_id"]] = [(float(e["onset"]), float(e["offset"])) for e in r["events"]]
    return out


def overlaps(span, events):
    return any(min(span[1], float(e["offset"])) > max(span[0], float(e["onset"])) for e in events)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--votes", type=int, default=3)
    ap.add_argument("--add-votes", type=int, default=5)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--min-iou", type=float, default=0.3)
    ap.add_argument("--max-shift", type=float, default=0.5)
    ap.add_argument("--out", default=f"{DATA}/splits/train_v29_pl2.jsonl")
    ap.add_argument("--teachers", nargs="+", default=TEACHERS, help="members whose nat + epnraw routes are fused")
    ap.add_argument("--pl-dir", default=W)
    a = ap.parse_args()
    routes = {p: [load(f"{a.pl_dir}/{m}_{rt}_{p}/predictions.jsonl") for m in a.teachers for rt in ("nat", "epnraw")] for p in ("comp", "parent")}
    rows = [json.loads(l) for l in open(SRC)]
    n = collections.Counter(); ev_in = collections.Counter(); ev_out = collections.Counter(); moved = added = 0
    with open(a.out, "w") as fh:
        for r in rows:
            t = r["tier"]; n[t] += 1; ev_in[t] += len(r["events"])
            if t in ("gold", "bronze"):
                fh.write(json.dumps(r) + "\n"); ev_out[t] += len(r["events"]); continue
            pool = "parent" if t == "parent_pseudo" else "comp"
            spans = [m.get(r["clip_id"], []) for m in routes[pool]]
            fused = fuse(spans, a.iou, a.votes, agg="median", fallback=True)
            if t == "silver":
                events, k = move_boundaries(r["events"], fused, a.min_iou, a.max_shift); moved += k
                for s, e in fuse(spans, a.iou, a.add_votes, agg="median", fallback=False):
                    if not overlaps((s, e), events):
                        events.append({"onset": round(s, 4), "offset": round(e, 4), "pl_added": True}); added += 1
                r = dict(r, events=sorted(events, key=lambda e: float(e["onset"])))
            else:
                r = dict(r, events=[{"onset": round(s, 3), "offset": round(e, 3)} for s, e in fused], pl_source=t)
            fh.write(json.dumps(r) + "\n"); ev_out[t] += len(r["events"])
    print(f"wrote {a.out}")
    for t in n:
        print(f"  {t:14} rows {n[t]:6d}  events {ev_in[t]:6d} -> {ev_out[t]:6d}")
    print(f"  silver edges moved {moved}, silver events added at >= {a.add_votes} votes {added}")


if __name__ == "__main__":
    main()
