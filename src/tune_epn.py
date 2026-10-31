"""Tune the proposal decode of an end-to-end EPN checkpoint on the natural validation clips.

Writes a decode config for predict.py whose "epn" block switches it to proposal selection;
the base keys are carried from the shipped natural config so the file stays self-contained.
"""
from __future__ import annotations
from paths import DATA, PRETRAINED, WORK  # noqa: F401

import argparse
import os
import json
import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import VaaniFrameDataset, collate
from epn_eval import run_eval
from model import tta_forward, load_checkpoint


@torch.no_grad()
def cache_outputs(ckpt, manifest, audio_root, device, batch_size=16, workers=8):
    model, _ = load_checkpoint(ckpt, device, use_ema=True)
    ds = VaaniFrameDataset(manifest, audio_root, max_sec=None, train=False)
    ds.records = [r for r in ds.records if not r.get("synthetic")]
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                    collate_fn=collate, pin_memory=True)
    probs, bnd, dist = {}, {}, {}
    for batch in dl:
        wav = batch["wav"].to(device, non_blocking=True)
        n_fr = int(batch["frame_mask"].shape[1])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = (tta_forward(model, wav, n_fr) if os.environ.get('TTA_SHIFT') == '1' else model(wav, n_fr))
        p = torch.sigmoid(out[0].float()).cpu().numpy()
        pon = torch.sigmoid(out[2].float()).cpu().numpy()
        poff = torch.sigmoid(out[3].float()).cpu().numpy()
        d = out[5].float().cpu().numpy()
        lengths = batch["frame_mask"].sum(dim=1).long().numpy()
        for j, cid in enumerate(batch["clip_ids"]):
            k = lengths[j]
            probs[cid] = p[j, :k]
            bnd[cid] = (pon[j, :k], poff[j, :k])
            dist[cid] = d[j, :k]
    return probs, bnd, dist, ds.records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val", default=f"{DATA}/splits/val_mixed.jsonl")
    ap.add_argument("--audio-root", default=f"{DATA}/prepared")
    ap.add_argument("--config", default=f"{WORK}/postproc_v11_natural.json")
    ap.add_argument("--refiner", default=f"{WORK}/refiner_v11.pt")
    ap.add_argument("--cache", default=None, help="pickle of cached outputs (read if present)")
    ap.add_argument("--out", required=True, help="decode config json for predict.py")
    args = ap.parse_args()
    device = "cuda"

    if args.cache and Path(args.cache).exists():
        probs, bnd, dist, records = pickle.load(open(args.cache, "rb"))
    else:
        probs, bnd, dist, records = cache_outputs(args.ckpt, args.val, args.audio_root, device)
        if args.cache:
            pickle.dump((probs, bnd, dist, records), open(args.cache, "wb"), protocol=4)
    refs = {r["clip_id"]: [(e["onset"], e["offset"]) for e in r["events"]] for r in records}
    durs = {r["clip_id"]: r["duration"] for r in records}
    spk = {r["clip_id"]: r.get("speaker", r["clip_id"]) for r in records}
    print(f"{len(refs)} natural validation clips")

    cfg = json.load(open(args.config))["config"]
    from boundary_refiner import load_refiner
    refiner = load_refiner(args.refiner, device)
    refiner.eval()
    res = run_eval(dist, probs, bnd, refs, durs, spk, cfg, refiner, device)

    best = res["best_full"]
    epn = {k: best[k] for k in ("p_min", "conf_min", "min_dur", "vote", "refine",
                                "suppress_accepted_only", "extend_min")}
    epn["k"] = 20
    out_cfg = {**cfg, "epn": epn}
    json.dump({"config": out_cfg, "natural_scores": {k: best[k] for k in ("f1", "dice", "combined")},
               "baseline": res["baseline"], "oof_gains": res["oof_gains"],
               "oof_chosen": res["oof_chosen"], "ckpt": args.ckpt},
              open(args.out, "w"), indent=2)
    print(f"wrote {args.out}: {json.dumps(epn)}")


if __name__ == "__main__":
    main()
