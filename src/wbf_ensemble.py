"""Proposal-level ensembling (weighted boxes fusion) across trained models.

Curve averaging blurs the activity edge and loses Event-F1 (dead-end #4).  Fusing each
model's decoded *events* instead keeps edges sharp: events from different models that
overlap above `iou` form a cluster, the cluster's edges are the vote-weighted mean, and a
cluster is emitted only if at least `min_votes` models agree.  Evaluated on natural
validation from the models' cached probabilities, speaker-disjoint 2-fold OOF.
"""
from __future__ import annotations
from paths import DATA, PRETRAINED, WORK  # noqa: F401

import argparse
import os
import json

import numpy as np
import torch

from boundary_refiner import BoundaryRefiner, refine_spans
from ensemble import load_cache
from epn_eval import DEC
from model import decode, snap_to_boundaries
from scorer import score


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def fuse(model_spans: list[list[tuple[float, float]]], thr: float, min_votes: int,
         weights=None, agg: str = "mean", fallback: bool = False, with_votes: bool = False) -> list[tuple[float, float]]:
    """model_spans[m] = that model's events; returns fused events.  `agg` sets how a cluster's edges
    are combined (mean, or the median of the voters); `fallback` keeps the largest cluster when no
    cluster reaches min_votes, since a silent clip scores Dice 0 whenever a reference exists."""
    clusters: list[dict] = []
    for m, spans in enumerate(model_spans):
        w = 1.0 if weights is None else weights[m]
        for s in spans:
            best, best_iou = None, thr
            for c in clusters:
                v = iou(s, c["box"])
                if v >= best_iou and m not in c["members"]:
                    best, best_iou = c, v
            if best is None:
                clusters.append({"sum": [w * s[0], w * s[1]], "w": w, "box": s, "members": {m}, "on": [s[0]], "off": [s[1]]})
            else:
                best["sum"][0] += w * s[0]
                best["sum"][1] += w * s[1]
                best["w"] += w
                best["members"].add(m)
                best["on"].append(s[0]); best["off"].append(s[1])
                best["box"] = (best["sum"][0] / best["w"], best["sum"][1] / best["w"])
    def edges(c):
        if agg == "median":
            import numpy as _np
            return (float(_np.median(c["on"])), float(_np.median(c["off"])))
        return c["box"]
    keep = [c for c in clusters if len(c["members"]) >= min_votes]
    if not keep and fallback and clusters:
        keep = [max(clusters, key=lambda c: len(c["members"]))]
    if with_votes:
        return sorted((round(edges(c)[0], 4), round(edges(c)[1], 4), len(c["members"])) for c in keep)
    out = [(round(edges(c)[0], 4), round(edges(c)[1], 4)) for c in keep]
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--versions", nargs="+", default=["v11", "v12", "v13", "v14", "v17"])
    ap.add_argument("--out", default=f"{WORK}/wbf_result.json")
    ap.add_argument("--dump", default=None, help="pickle the per-member val spans, refs, durations and speakers for downstream rescoring")
    args = ap.parse_args()
    device = "cuda"
    torch.set_num_threads(4)

    per_model, refs, durs, spk = {}, None, None, None
    for v in args.versions:
        if v.endswith(":epn") or v.endswith(":epnraw"):
            # proposal-route member from a cache_val_outputs.py pickle; ":epnraw" skips the refiner
            # and the long-span extension (the refiner is trained on threshold-route edges)
            import pickle
            from epn_eval import extend_long_spans
            from epn_head import epn_decode
            raw = v.endswith(":epnraw")
            base_v = v[:-7] if raw else v[:-4]
            blob = pickle.load(open(f"{WORK}/val_outputs_{base_v}{os.environ.get('CACHE_TAG', '')}.pkl", "rb"))
            outs, records = blob["outputs"], blob["records"]
            if refs is None:
                refs = {r["clip_id"]: [(e["onset"], e["offset"]) for e in r["events"]] for r in records}
                durs = {r["clip_id"]: r["duration"] for r in records}
                spk = {r["clip_id"]: r.get("speaker", r["clip_id"]) for r in records}
            cfg = json.load(open(f"{WORK}/postproc_{base_v}_natural.json"))["config"]
            base = {k: x for k, x in cfg.items() if k in DEC}
            ecfg = json.load(open(f"{WORK}/postproc_{base_v}_epnroute.json"))["config"]["epn"]
            refiner = BoundaryRefiner().to(device)
            refiner.load_state_dict(torch.load(f"{WORK}/refiner_{base_v}.pt", map_location="cpu")["model"])
            refiner.eval()
            spans = {}
            for cid in refs:
                o = outs[cid]
                e = epn_decode(o["act"], o["dist"][:, 0], o["dist"][:, 1], p_min=ecfg["p_min"], k=20,
                               conf_min=ecfg["conf_min"], duration=durs[cid], vote=ecfg["vote"],
                               suppress_accepted_only=ecfg["suppress_accepted_only"])
                if raw:
                    spans[cid] = sorted(e)
                    continue
                if ecfg.get("extend_min"):
                    e = extend_long_spans(e, decode(o["act"], duration=durs[cid], **base), ecfg["extend_min"])
                spans[cid] = refine_spans(e, o["act"], o["on"], o["off"], refiner, device, durs[cid])
            per_model[v] = spans
            s = score(refs, spans, durs)
            print(f"{v}: F1 {s.event_f1:.4f} Dice {s.segment_dice:.4f} Comb {s.combined:.4f}", flush=True)
            continue
        probs, durations, records, bnd = load_cache(f"{WORK}/val_mixed_probs_{v}{os.environ.get('CACHE_TAG', '')}.pkl")
        nat = [r for r in records if not r.get("synthetic")]
        if refs is None:
            refs = {r["clip_id"]: [(e["onset"], e["offset"]) for e in r["events"]] for r in nat}
            durs = {r["clip_id"]: durations[r["clip_id"]] for r in nat}
            spk = {r["clip_id"]: r.get("speaker", r["clip_id"]) for r in nat}
        cfg = json.load(open(f"{WORK}/postproc_{v}_natural{os.environ.get('NAT_CFG_TAG', '')}.json"))["config"]
        base = {k: x for k, x in cfg.items() if k in DEC}
        snap = cfg.get("snap_sec", 0.0)
        refiner = BoundaryRefiner().to(device)
        refiner.load_state_dict(torch.load(f"{WORK}/refiner_{v}.pt", map_location="cpu")["model"])
        refiner.eval()
        spans = {}
        for cid in refs:
            pol_dir = os.environ.get("POLICY_DIR")
            if pol_dir and os.path.exists(f"{pol_dir}/policy_{v}.pkl"):
                import pickle as _pk; from clip_policy import apply as _apply
                if not hasattr(main, "_pol"): main._pol = {}
                if v not in main._pol: main._pol[v] = _pk.load(open(f"{pol_dir}/policy_{v}.pkl", "rb"))
                base = dict(base, threshold=_apply(main._pol[v], probs[cid], durs[cid]))
            s = decode(probs[cid], duration=durs[cid], **base)
            if snap:
                s = snap_to_boundaries(s, bnd[cid][0], bnd[cid][1], search_sec=snap)
            spans[cid] = refine_spans(s, probs[cid], bnd[cid][0], bnd[cid][1], refiner, device,
                                      durs[cid])
        per_model[v] = spans
        s = score(refs, spans, durs)
        print(f"{v}: F1 {s.event_f1:.4f} Dice {s.segment_dice:.4f} Comb {s.combined:.4f}", flush=True)
    ids = list(refs)
    if args.dump:
        import pickle
        pickle.dump({"per_model": per_model, "refs": refs, "durs": durs, "spk": spk, "versions": args.versions}, open(args.dump, "wb"))
        print(f"dumped per-member spans to {args.dump}", flush=True)

    def sc(cids, preds):
        return score({c: refs[c] for c in cids}, {c: preds[c] for c in cids},
                     {c: durs[c] for c in cids})

    order_models = list(args.versions)
    n = len(order_models)
    votes = (1, 2, 3) if n <= 6 else tuple(range(2, max(6, n // 2 + 2)))
    grid = [dict(iou=t, min_votes=mv) for t in (0.5, 0.7) for mv in votes]
    cache, rows = {}, []
    for i, g in enumerate(grid):
        cache[i] = {cid: fuse([per_model[v][cid] for v in order_models], g["iou"], g["min_votes"])
                    for cid in ids}
        s = sc(ids, cache[i])
        rows.append({**g, "f1": s.event_f1, "dice": s.segment_dice, "combined": s.combined,
                     "preds": s.n_pred_events})
        print(f"iou {g['iou']:.1f} votes>={g['min_votes']}: F1 {s.event_f1:.4f} Dice "
              f"{s.segment_dice:.4f} Comb {s.combined:.4f} preds {s.n_pred_events}", flush=True)

    best_single = max(order_models, key=lambda v: sc(ids, per_model[v]).combined)
    print(f"best single model on full val: {best_single}")
    speakers = sorted(set(spk.values()))
    rng = np.random.default_rng(0)
    rng.shuffle(speakers)
    half = set(speakers[: len(speakers) // 2])
    folds = [[c for c in ids if spk[c] in half], [c for c in ids if spk[c] not in half]]
    gains = []
    for f in (0, 1):
        fit, held = folds[1 - f], folds[f]
        bi = max(range(len(grid)), key=lambda i: sc(fit, cache[i]).combined)
        bv = max(order_models, key=lambda v: sc(fit, per_model[v]).combined)
        sb, se = sc(held, per_model[bv]), sc(held, cache[bi])
        gains.append(se.combined - sb.combined)
        print(f"fold {f}: chose {grid[bi]} vs best single {bv}: held-out {sb.combined:.4f} -> "
              f"{se.combined:.4f} ({se.combined - sb.combined:+.4f})  F1 {sb.event_f1:.4f}->"
              f"{se.event_f1:.4f} Dice {sb.segment_dice:.4f}->{se.segment_dice:.4f}")
    print(f"out-of-fold mean gain {np.mean(gains):+.4f} (folds {gains[0]:+.4f}, {gains[1]:+.4f}; "
          f"same sign {gains[0] * gains[1] > 0})")
    json.dump({"rows": rows, "oof_gains": gains}, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
