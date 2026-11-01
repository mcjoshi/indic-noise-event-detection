"""Vote ablation of Table III: single decodes vs votes over threshold, proposal and both decode types.

usage: python3 scripts/ablate_vote.py --manifest data/splits/val.jsonl --decodes DIR
DIR holds <run>_natural/predictions.jsonl and <run>_epnraw/predictions.jsonl for the five router-free models of
submission 210 (stx374, stx365, u373_mix, st137, st390), written by src/predict.py with configs/decode/.
Vote thresholds are chosen on one speaker half and scored on the other; intervals bootstrap speakers.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "inference")]
from infer import weighted_vote  # noqa: E402
from scorer import score  # noqa: E402

WEIGHTS = {"stx374": 0.5, "stx365": 0.5, "u373_mix": 0.5, "st137": 1.0, "st390": 0.5}


def load(path):
    return {(r := json.loads(line))["clip_id"]: [(e["onset"], e["offset"]) for e in r["events"]] for line in open(path)}


ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--decodes", required=True)
ap.add_argument("--boot", type=int, default=300)
a = ap.parse_args()

rows = [json.loads(line) for line in open(a.manifest)]
refs = {r["clip_id"]: [(e["onset"], e["offset"]) for e in r["events"]] for r in rows}
durs = {r["clip_id"]: r["duration"] for r in rows}
spk = {r["clip_id"]: r["speaker"] for r in rows}
clips = sorted(refs)
T = {m: load(f"{a.decodes}/{m}_natural/predictions.jsonl") for m in WEIGHTS}
P = {m: load(f"{a.decodes}/{m}_epnraw/predictions.jsonl") for m in WEIGHTS}


def sc(pred, cs):
    return score({c: refs[c] for c in cs}, {c: pred.get(c, []) for c in cs}, {c: durs[c] for c in cs}).combined


def fused(routes, ws, keep, long=False):
    f = {"iou": 0.6, "keep_if_summed_weight_at_least": keep}
    if long:
        f.update(long_sec=6.0, long_keep=1.5, iou_long=0.45)
    return {c: weighted_vote([r.get(c, []) for r in routes], ws, f) for c in clips}


speakers = sorted(set(spk.values()))
rng = np.random.default_rng(0)
rng.shuffle(speakers)
half_a = set(speakers[: len(speakers) // 2])
halves = [[c for c in clips if spk[c] in half_a], [c for c in clips if spk[c] not in half_a]]


def crossfit(routes, ws, grid):
    preds = {k: fused(routes, ws, k) for k in grid}
    picks = [max(grid, key=lambda k: sc(preds[k], halves[1 - h])) for h in (0, 1)]
    return float(np.mean([sc(preds[k], halves[h]) for h, k in enumerate(picks)])), preds[max(set(picks), key=picks.count)]


tw = list(WEIGHTS.values())
both = [r for m in WEIGHTS for r in (T[m], P[m])]
bw = [w for w in WEIGHTS.values() for _ in (0, 1)]
single_t = float(np.mean([sc(T[m], clips) for m in WEIGHTS]))
single_p = float(np.mean([sc(P[m], clips) for m in WEIGHTS]))
vote_t, _ = crossfit([T[m] for m in WEIGHTS], tw, [0.5, 1.0, 1.25, 1.5, 2.0, 2.5])
vote_p, pred_p = crossfit([P[m] for m in WEIGHTS], tw, [0.5, 1.0, 1.25, 1.5, 2.0, 2.5])
pred_both, pred_long = fused(both, bw, 2.5), fused(both, bw, 2.5, long=True)
vote_both = float(np.mean([sc(pred_both, h) for h in halves]))
vote_long = float(np.mean([sc(pred_long, h) for h in halves]))

by_spk = {}
for c in clips:
    by_spk.setdefault(spk[c], []).append(c)
d_both, d_long = [], []
for _ in range(a.boot):
    cs = [c for s in rng.choice(list(by_spk), len(by_spk)) for c in by_spk[s]]
    s_both = sc(pred_both, cs)
    d_both.append(s_both - sc(pred_p, cs))
    d_long.append(sc(pred_long, cs) - s_both)

print(f"{len(clips)} clips, {len(speakers)} speakers")
for name, v in [("single model, threshold decode (mean of 5)", single_t), ("single model, proposal decode (mean of 5)", single_p),
                ("vote over 5 threshold decodes", vote_t), ("vote over 5 proposal decodes", vote_p),
                ("vote over all 10 decodes (keep 2.5)", vote_both), ("all 10 decodes + long-span rule", vote_long)]:
    print(f"  {v:.4f}  {name}")
for name, d in [("both decode types vs proposal-only vote", d_both), ("long-span rule", d_long)]:
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"  {name}: {np.mean(d):+.4f} [{lo:+.4f}, {hi:+.4f}]")
