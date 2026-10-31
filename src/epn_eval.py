"""Score EPN proposal decoding on the natural validation clips against the shipped route.

Shared by the frozen-state fit (train_epn.py) and the end-to-end tuner (tune_epn.py): a grid
over the decode operating point, then the speaker-disjoint 2-fold out-of-fold gain against
decode+snap+refiner, both through the competition scorer.
"""
from __future__ import annotations

import numpy as np

from boundary_refiner import refine_spans
from epn_head import epn_decode
from model import decode, snap_to_boundaries
from scorer import score

DEC = {"threshold", "offset_threshold", "median_k", "merge_gap", "min_dur",
       "onset_bias", "offset_bias"}

# p_min 0.1 vs 0.3 and refine=False never changed the ranking in the frozen-state sweep
GRID = [dict(p_min=0.3, conf_min=cm, min_dur=0.0, vote=v, refine=True,
             suppress_accepted_only=sa, extend_min=ex)
        for cm in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        for v in (False, True)
        for sa in (False, True)
        for ex in (None, 1.0, 0.5)]


def extend_long_spans(spans, runs, min_dur):
    """Long proposals take their extent from the activity runs they overlap: the IoU loss
    weights a 3 s event like a 0.1 s one, so long proposals come out short and cost Dice."""
    out = []
    for a, b in spans:
        if b - a >= min_dur:
            for ra, rb in runs:
                if ra < b and rb > a:
                    a, b = min(a, ra), max(b, rb)
        out.append((round(a, 4), round(b, 4)))
    return out


def run_eval(dist, probs, bnd, refs, durs, spk, cfg, refiner, device, grid=GRID,
             baseline=None) -> dict:
    base = {k: v for k, v in cfg.items() if k in DEC}
    snap = cfg.get("snap_sec", 0.0)
    if baseline is None:
        baseline = {}
        for cid in refs:
            s = decode(probs[cid], duration=durs[cid], **base)
            if snap:
                s = snap_to_boundaries(s, bnd[cid][0], bnd[cid][1], search_sec=snap)
            baseline[cid] = refine_spans(s, probs[cid], bnd[cid][0], bnd[cid][1], refiner,
                                         device, durs[cid])

    def sc(cids, preds):
        return score({c: refs[c] for c in cids}, {c: preds[c] for c in cids},
                     {c: durs[c] for c in cids})

    runs = {cid: decode(probs[cid], duration=durs[cid], **base) for cid in refs}

    def build(cids, g):
        out = {}
        for cid in cids:
            d = dist[cid]
            s = epn_decode(probs[cid], d[:, 0], d[:, 1], p_min=g["p_min"], k=20,
                           conf_min=g["conf_min"], min_dur=g["min_dur"], duration=durs[cid],
                           vote=g["vote"], suppress_accepted_only=g["suppress_accepted_only"])
            if g.get("extend_min"):
                s = extend_long_spans(s, runs[cid], g["extend_min"])
            if g["refine"]:
                s = refine_spans(s, probs[cid], bnd[cid][0], bnd[cid][1], refiner, device,
                                 durs[cid])
            out[cid] = s
        return out

    all_ids = list(refs)
    s0 = sc(all_ids, baseline)
    print(f"\nbaseline (decode+snap+refiner)  F1 {s0.event_f1:.4f}  Dice {s0.segment_dice:.4f}"
          f"  Combined {s0.combined:.4f}  preds {s0.n_pred_events}", flush=True)

    cache = {}
    rows = []
    for i, g in enumerate(grid):
        cache[i] = build(all_ids, g)
        s = sc(all_ids, cache[i])
        rows.append({**g, "f1": s.event_f1, "dice": s.segment_dice, "combined": s.combined,
                     "preds": s.n_pred_events})
    order = sorted(range(len(rows)), key=lambda i: -rows[i]["combined"])
    print(f"{'conf':>6}{'vote':>6}{'sacc':>6}{'ext':>6}{'F1':>9}{'Dice':>9}{'Comb':>9}{'preds':>7}")
    for i in order[:12]:
        r = rows[i]
        print(f"{r['conf_min']:>6.2f}{str(r['vote']):>6}{str(r['suppress_accepted_only']):>6}"
              f"{str(r['extend_min']):>6}{r['f1']:>9.4f}{r['dice']:>9.4f}"
              f"{r['combined']:>9.4f}{r['preds']:>7}")

    speakers = sorted(set(spk[c] for c in refs))
    rng = np.random.default_rng(0)
    rng.shuffle(speakers)
    half = set(speakers[: len(speakers) // 2])
    folds = [[c for c in refs if spk[c] in half], [c for c in refs if spk[c] not in half]]
    gains, chosen = [], []
    for f in (0, 1):
        fit, held = folds[1 - f], folds[f]
        bi = max(range(len(grid)), key=lambda i: sc(fit, cache[i]).combined)
        sb = sc(held, baseline)
        se = sc(held, cache[bi])
        gains.append(se.combined - sb.combined)
        chosen.append(grid[bi])
        print(f"fold {f}: chose {grid[bi]}\n  held-out baseline F1 {sb.event_f1:.4f} Dice "
              f"{sb.segment_dice:.4f} Comb {sb.combined:.4f}\n  held-out EPN      F1 "
              f"{se.event_f1:.4f} Dice {se.segment_dice:.4f} Comb {se.combined:.4f} "
              f"({se.combined - sb.combined:+.4f})", flush=True)
    print(f"\nout-of-fold mean gain {float(np.mean(gains)):+.4f} (folds {gains[0]:+.4f}, "
          f"{gains[1]:+.4f}; same sign {gains[0] * gains[1] > 0})")
    best = rows[order[0]]
    return {"baseline": s0.as_dict(), "rows": [rows[i] for i in order], "best_full": best,
            "oof_gains": gains, "oof_chosen": chosen}
