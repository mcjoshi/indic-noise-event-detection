"""Student training set for the unified distillation run. Natural rows that carry teacher-derived labels (transcribed
Vaani, Bronze) are relabelled from the 198 natural teachers' decodes, conditioned on the transcripts' event tags
(label_control.py: 1.2578 on Gold val vs 1.2415 for the old teacher); Gold and competition Silver keep their labels;
wider-corpus parent_pseudo rows are dropped; synthetic rows keep their exact labels and are sampled 1:1 with natural.
usage: python3 build_student_mix.py [--out data/splits/train_student_mix.jsonl] [--teach M1,M2,...]"""
from __future__ import annotations
from paths import DATA, PRETRAINED, WORK  # noqa: F401

import json
import os
import random
import sys

from pl_transcript import iou, MAN
from transcript_labels import label
from wbf_ensemble import fuse

W = str(WORK)
TEACH = ["v373_whisper_unf_tp", "v374_whisper_unf_tp2", "v365_whisper_unf_full", "v137_n3_longunf", "v161_v145_dice3_s2"]
if "--teach" in sys.argv:
    TEACH = sys.argv[sys.argv.index("--teach") + 1].split(",")  # e.g. the 211 teacher: sty373,st374,v365_whisper_unf_full,v137_n3_longunf,v161_v145_dice3_s2
OLD = {"v137_n3_longunf", "v161_v145_dice3_s2"}
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else f"{DATA}/splits/train_student_mix.jsonl"
EXT = int(sys.argv[sys.argv.index("--ext") + 1]) if "--ext" in sys.argv else 0
NAT_CAP = int(sys.argv[sys.argv.index("--natural") + 1]) if "--natural" in sys.argv else 0
SYN_RATIO = float(sys.argv[sys.argv.index("--syn-ratio") + 1]) if "--syn-ratio" in sys.argv else 1.0
EXT_POOLS = sys.argv[sys.argv.index("--extpools") + 1].split(",") if "--extpools" in sys.argv else ["ext"]


def load(paths):
    d = {}
    for p in paths:
        if os.path.exists(p):
            for l in open(p):
                r = json.loads(l); d[r["clip_id"]] = [(float(e["onset"]), float(e["offset"])) for e in r["events"]]
    return d


def routes(pool):
    out = []
    for m in TEACH:
        for rt in ("nat", "epnraw"):
            if m in OLD:
                ps = [f"{W}/pl_tp/{m}_{rt}_tp/predictions.jsonl", f"{W}/pl_tp/{m}_{rt}_tp2/predictions.jsonl"] if pool == "tp" or pool.startswith("ext") else [f"{W}/pl/{m}_{rt}_comp/predictions.jsonl"]
            else:
                ps = [f"{W}/pl198/{m}_{rt}_{pool}/predictions.jsonl"]
            out.append(load(ps))
    return out


R = {"tp": routes("tp"), "comp": routes("comp")}
if EXT:
    for xp in EXT_POOLS:
        R[xp] = routes(xp)
tr = {}
for p in (f"{W}/pl_tp/manifest_r1.jsonl", f"{W}/pl_tp/manifest_r2.jsonl", MAN):
    for l in open(p):
        r = json.loads(l)
        if (r.get("transcript") or "").strip():
            tr.setdefault(r["clip_id"], r["transcript"])
nat, syn, stat = [], [], {"kept": 0, "relabelled": 0, "empty": 0, "no_teacher": 0, "dropped_parent": 0}
n = 2 * len(TEACH)
for l in open(f"{DATA}/splits/train_unified_mix.jsonl"):
    r = json.loads(l); c = r["clip_id"]; w = r["wav"]
    if "/synth" in w or "/official" in w:
        syn.append(r); continue
    if "parent_prepared" in w:
        stat["dropped_parent"] += 1; continue
    if r["tier"] == "gold" or (r["tier"] == "silver" and w.startswith("wav/")):
        nat.append(r); stat["kept"] += 1; continue
    pool = "tp" if "/vaani_tp/" in w else "comp"
    rs = R[pool]
    if c not in tr or not all(c in m for m in rs):
        nat.append(r); stat["no_teacher"] += 1; continue
    cands = []
    for v in range(n, 0, -1):
        for s in fuse([m[c] for m in rs], 0.6, v, agg="median", fallback=False):
            if all(iou(s, x[:2]) < 0.3 for x in cands):
                cands.append((s[0], s[1], v))
    evs = label(tr[c], float(r["duration"]), cands, n, w_pos=3.0, fill=False)
    if not evs:
        stat["empty"] += 1; continue
    nat.append(dict(r, events=[{"onset": round(s, 3), "offset": round(e, 3)} for s, e in evs], pl_source="teacher198_transcript"))
    stat["relabelled"] += 1
if EXT:
    man = {}
    for p in (f"{W}/pl_tp/manifest_r1.jsonl", f"{W}/pl_tp/manifest_r2.jsonl"):
        for l in open(p):
            r = json.loads(l); man[r["clip_id"]] = r
    added = 0
    ext_ids = [(xp, c) for xp in EXT_POOLS for c in open(f"{W}/pl198/ids_{xp}.txt").read().split()][:EXT]
    for xp, c in ext_ids:
        rs = R[xp]
        if c not in man or c not in tr or not all(c in m for m in rs):
            stat["no_teacher"] += 1; continue
        cands = []
        for v in range(n, 0, -1):
            for s_ in fuse([m[c] for m in rs], 0.6, v, agg="median", fallback=False):
                if all(iou(s_, x[:2]) < 0.3 for x in cands):
                    cands.append((s_[0], s_[1], v))
        evs = label(tr[c], float(man[c]["duration"]), cands, n, w_pos=3.0, fill=False)
        if not evs:
            stat["empty"] += 1; continue
        nat.append({"clip_id": c, "wav": man[c]["wav"], "duration": man[c]["duration"], "tier": "silver", "pl_source": "teacher198_transcript",
                    "events": [{"onset": round(a, 3), "offset": round(b, 3)} for a, b in evs]})
        added += 1
    stat["ext_added"] = added
if NAT_CAP and len(nat) > NAT_CAP:
    keep = [r for r in nat if r.get("pl_source") != "teacher198_transcript"]
    rel = [r for r in nat if r.get("pl_source") == "teacher198_transcript"]
    random.Random(2).shuffle(rel)
    nat = keep + rel[: max(0, NAT_CAP - len(keep))]
random.Random(0).shuffle(syn)
syn.sort(key=lambda r: "/official" not in r["wav"])
n_syn = int(len(nat) * SYN_RATIO)
syn = (syn * (n_syn // max(1, len(syn)) + 1))[:n_syn]
rows = nat + syn
random.Random(1).shuffle(rows)
with open(OUT, "w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
print(f"wrote {OUT}: natural {len(nat)} {stat}, synthetic {len(syn)} (exact labels, repeated if needed)")
