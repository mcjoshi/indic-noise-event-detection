"""Stage-3 manifest: Bronze clips (transcripts, no timestamps) get transcript-conditioned labels instead of the
teacher's raw spans. Candidates = the round-1 teacher's 6 routes (v137, v161, v273 x threshold/proposal, already in
work/pl/*_comp) fused at every vote level; one span per transcript event tag, Hungarian-matched on the tag's position
prior (w_pos 3) and vote support, no fill (val control: 1.2403 vs the teacher's 1.1903 on Gold). Every other row is
copied from train_v29_pl2.jsonl unchanged.
usage: python3 pl_transcript.py [--src data/splits/train_v29_pl2.jsonl] [--out data/splits/train_v29_pl4.jsonl]
"""
from __future__ import annotations
from paths import DATA, PRETRAINED, WORK  # noqa: F401

import argparse
import collections
import json

from transcript_labels import label
from wbf_ensemble import fuse

TEACHERS = ["v137_n3_longunf", "v161_v145_dice3_s2", "v273_hubert_unf"]
MAN = f"{DATA}/prepared/manifest.jsonl"


def load(path):
    return {r["clip_id"]: [(float(e["onset"]), float(e["offset"])) for e in r["events"]] for r in (json.loads(l) for l in open(path) if l.strip())}


def iou(a, b):
    i = max(0.0, min(a[1], b[1]) - max(a[0], b[0])); u = max(a[1], b[1]) - min(a[0], b[0])
    return i / u if u > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=f"{DATA}/splits/train_v29_pl2.jsonl")
    ap.add_argument("--out", default=f"{DATA}/splits/train_v29_pl4.jsonl")
    ap.add_argument("--w-pos", type=float, default=3.0)
    ap.add_argument("--tiers", nargs="+", default=["bronze_pseudo", "bronze"])
    a = ap.parse_args()
    routes = [load(f"{WORK}/pl/{m}_{rt}_comp/predictions.jsonl") for m in TEACHERS for rt in ("nat", "epnraw")]
    man = {r["clip_id"]: r for r in (json.loads(l) for l in open(MAN))}
    n = collections.Counter(); ev = collections.Counter()
    with open(a.out, "w") as fh:
        for line in open(a.src):
            r = json.loads(line)
            c = r["clip_id"]
            if r["tier"] in a.tiers and c in man and (man[c].get("transcript") or "").strip() and all(c in m for m in routes):
                spans = [m[c] for m in routes]; dur = float(man[c]["duration"])
                cands = []
                for v in range(len(routes), 0, -1):
                    for s in fuse(spans, 0.6, v, agg="median", fallback=False):
                        if all(iou(s, x[:2]) < 0.3 for x in cands):
                            cands.append((s[0], s[1], v))
                evs = label(man[c]["transcript"], dur, cands, len(routes), w_pos=a.w_pos, fill=False)
                ev[r["tier"], "before"] += len(r["events"]); ev[r["tier"], "after"] += len(evs)
                r = dict(r, events=[{"onset": round(s, 3), "offset": round(e, 3)} for s, e in evs], tier="bronze_pseudo", pl_source="transcript")
                n["relabelled"] += 1
            else:
                n["copied"] += 1
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {a.out}: {dict(n)}; events {dict(ev)}")


if __name__ == "__main__":
    main()
