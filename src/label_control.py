"""Label-quality control: transcript-conditioned labels built from a teacher's val routes, scored against Gold val.
usage: python3 label_control.py MEMBER,MEMBER,..."""
from __future__ import annotations
from paths import DATA, PRETRAINED, WORK  # noqa: F401

import json
import pickle
import sys

from scorer import score  # noqa: E402
from transcript_labels import label  # noqa: E402
from wbf_ensemble import fuse  # noqa: E402
from pl_transcript import iou, MAN  # noqa: E402

W = str(WORK)
man = {r["clip_id"]: r for r in (json.loads(l) for l in open(MAN))}
for arg in sys.argv[1:]:
    mem = arg.split(",")
    per, refs, durs = {}, None, None
    for m in mem:
        only = m.endswith("!"); m = m.rstrip("!")
        b = pickle.load(open(f"{W}/probe_spans_ct_{m}_val.pkl", "rb"))
        for r in ((m,) if only else (m, f"{m}:epnraw")):
            per[r] = b["per_model"][r]
        refs, durs = b["refs"], b["durs"]
    routes = list(per.values()); n = len(routes)
    clips = [c for c in refs if (man.get(c, {}).get("transcript") or "").strip()]
    lab, raw = {}, {}
    for c in clips:
        spans = [r[c] for r in routes]
        cands = []
        for v in range(n, 0, -1):
            for s in fuse(spans, 0.6, v, agg="median", fallback=False):
                if all(iou(s, x[:2]) < 0.3 for x in cands):
                    cands.append((s[0], s[1], v))
        lab[c] = label(man[c]["transcript"], float(durs[c]), cands, n, w_pos=3.0, fill=False)
    R = {c: refs[c] for c in clips}; D = {c: durs[c] for c in clips}
    s = score(R, lab, D)
    best = max(((v, score(R, {c: fuse([r[c] for r in routes], 0.6, v, agg="median", fallback=True) for c in clips}, D)) for v in range(1, n + 1)), key=lambda x: x[1].event_f1 + x[1].segment_dice)
    print(f"{mem}: clips {len(clips)} | transcript labels F1 {s.event_f1:.4f} Dice {s.segment_dice:.4f} Combined {s.event_f1 + s.segment_dice:.4f} | best raw vote {best[0]} of {n}: {best[1].event_f1 + best[1].segment_dice:.4f}", flush=True)
