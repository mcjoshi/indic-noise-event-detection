"""Label the ingested transcribed-Vaani clips from the teacher's six routes + transcripts, and write
train_v29_pl4_tp.jsonl = train_v29_pl4 + these rows (tier silver)."""
from __future__ import annotations
from paths import DATA, PRETRAINED, WORK  # noqa: F401

import json

from transcript_labels import label
from wbf_ensemble import fuse

W, R = f"{WORK}/pl_tp", f"{DATA}/vaani_tp"
TEACHERS = ["v137_n3_longunf", "v161_v145_dice3_s2", "v273_hubert_unf"]


def load(path):
    return {r["clip_id"]: [(float(e["onset"]), float(e["offset"])) for e in r["events"]] for r in (json.loads(l) for l in open(path) if l.strip())}


def iou(a, b):
    i = max(0.0, min(a[1], b[1]) - max(a[0], b[0])); u = max(a[1], b[1]) - min(a[0], b[0])
    return i / u if u > 0 else 0.0


import sys
RND = int(sys.argv[sys.argv.index("--round") + 1]) if "--round" in sys.argv else 1
SFX, MAN, SRC, OUT = {1: ("tp", "manifest_r1", "train_v29_pl4", "train_v29_pl4_tp"), 2: ("tp2", "manifest_r2", "train_v29_pl4_tp", "train_v29_pl4_tp2")}[RND]
routes = [load(f"{W}/{m}_{rt}_{SFX}/predictions.jsonl") for m in TEACHERS for rt in ("nat", "epnraw")]
n = ev = empty = 0
with open(f"{DATA}/splits/{OUT}.jsonl", "w") as fh:
    for line in open(f"{DATA}/splits/{SRC}.jsonl"):
        fh.write(line)
    for line in open(f"{W}/{MAN}.jsonl"):
        r = json.loads(line); c = r["clip_id"]
        if not all(c in m for m in routes):
            continue
        cands = []
        for v in range(6, 0, -1):
            for s in fuse([m[c] for m in routes], 0.6, v, agg="median", fallback=False):
                if all(iou(s, x[:2]) < 0.3 for x in cands):
                    cands.append((s[0], s[1], v))
        evs = label(r["transcript"], r["duration"], cands, 6, w_pos=3.0, fill=False)
        if not evs:
            empty += 1; continue
        fh.write(json.dumps({"clip_id": c, "wav": r["wav"], "duration": r["duration"], "tier": "silver", "pl_source": "vaani_tp",
                             "events": [{"onset": round(s, 3), "offset": round(e, 3)} for s, e in evs]}) + "\n")
        n += 1; ev += len(evs)
print(f"wrote {OUT}.jsonl: +{n} transcribed-Vaani clips, {ev} events; {empty} clips without a matched candidate skipped")
