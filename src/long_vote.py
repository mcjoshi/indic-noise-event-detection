"""Duration-aware vote for the routed natural half: clusters whose median span is at least LONG seconds are kept at
LONG_VOTES votes instead of VOTES (long references are found by the threshold routes but the EPN routes fragment them,
so they rarely reach the full vote). Same clustering as wbf_ensemble.fuse (IoU 0.6, running-mean box, median edges,
largest-cluster fallback). usage: python3 long_vote.py NAME LONG LONG_VOTES MEMBER...   (LONG 0 reproduces build_small; IOU_LONG=0.45 relaxes the cluster IoU for long spans)"""
from __future__ import annotations
from paths import DATA, PRETRAINED, WORK  # noqa: F401

import json
import os
import sys

import numpy as np

from wbf_ensemble import iou  # noqa: E402

W = str(WORK)


def long_fuse(spans_by_route, votes, long_sec, long_votes, iou_long=0.0):
    cl = []
    for ri, spans in enumerate(spans_by_route):
        for s in spans:
            best, bv = None, (iou_long if iou_long and s[1] - s[0] >= long_sec else 0.6)
            for k in cl:
                v = iou(s, k["box"])
                if v >= bv and ri not in k["src"]:
                    best, bv = k, v
            if best is None:
                cl.append({"box": s, "src": {ri: s}})
            else:
                best["src"][ri] = s
                best["box"] = (float(np.mean([x[0] for x in best["src"].values()])), float(np.mean([x[1] for x in best["src"].values()])))
    out = []
    for k in cl:
        med = (float(np.median([v[0] for v in k["src"].values()])), float(np.median([v[1] for v in k["src"].values()])))
        need = long_votes if long_sec and med[1] - med[0] >= long_sec else votes
        if len(k["src"]) >= need:
            out.append(med)
    if not out and cl:
        k = max(cl, key=lambda k: len(k["src"]))
        out = [(float(np.median([v[0] for v in k["src"].values()])), float(np.median([v[1] for v in k["src"].values()])))]
    return sorted((round(a, 4), round(b, 4)) for a, b in out)


if __name__ == "__main__":
    name, long_sec, long_votes = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    iou_long = float(os.environ.get("IOU_LONG", "0")); BASE_VOTES = int(os.environ.get("BASE_VOTES", "4")); members = sys.argv[4:]
    routes = []
    for m in members:
        for suf in ("nat", "epnraw"):
            routes.append({r["clip_id"]: [(e["onset"], e["offset"]) for e in r["events"]] for r in map(json.loads, open(f"{W}/submission_{m}{suf}/predictions.jsonl"))})
    d = f"{W}/sub_{name}"; os.makedirs(d, exist_ok=True)
    with open(f"{d}/natural_fused.jsonl", "w") as fh:
        for c in sorted(routes[0]):
            ev = long_fuse([r[c] for r in routes], BASE_VOTES, long_sec, long_votes, iou_long)
            fh.write(json.dumps({"clip_id": c, "events": [{"onset": a, "offset": b} for a, b in ev]}) + "\n")
    print(f"LONG_VOTE {name}: long >= {long_sec}s at {long_votes} votes; natural events {sum(len(long_fuse([r[c] for r in routes], BASE_VOTES, long_sec, long_votes, iou_long)) for c in routes[0])}")
