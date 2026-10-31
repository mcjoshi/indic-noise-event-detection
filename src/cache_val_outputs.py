"""Cache every head of a checkpoint on the natural validation clips: activity, onset, offset,
per-category probabilities and proposal distances, plus the clip records."""
from __future__ import annotations
from paths import DATA, PRETRAINED, WORK  # noqa: F401

import argparse
import os
import pickle

import torch
from torch.utils.data import DataLoader

from dataset import VaaniFrameDataset, collate
from model import load_checkpoint, tta_forward


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val", default=f"{DATA}/splits/val_mixed.jsonl")
    ap.add_argument("--audio-root", default=f"{DATA}/prepared")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-synthetic", action="store_true")
    ap.add_argument("--residual-root", default=None)
    args = ap.parse_args()
    device = os.environ.get("VAANI_DEVICE", "cuda")  # "cpu" when every GPU is taken
    frac = float(os.environ.get("VAANI_MEM_FRAC", "0"))
    if frac > 0 and torch.cuda.is_available():
        # fail fast with an OOM traceback instead of deadlocking a shared card
        torch.cuda.set_per_process_memory_fraction(frac)
    model, _ = load_checkpoint(args.ckpt, device, use_ema=True)
    ds = VaaniFrameDataset(args.val, args.audio_root, max_sec=None, train=False, residual_root=args.residual_root)
    if not args.keep_synthetic:
        ds.records = [r for r in ds.records if not r.get("synthetic")]
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=8, collate_fn=collate,
                    pin_memory=True)
    out = {}
    for batch in dl:
        wav = batch["wav"].to(device, non_blocking=True)
        n_fr = int(batch["frame_mask"].shape[1])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
            res_t = batch["res"].to(device, non_blocking=True) if batch.get("res") is not None else None
            res = tta_forward(model, wav, n_fr, res_t) if os.environ.get("TTA_SHIFT") == "1" else model(wav, n_fr, res_t)
        act = torch.sigmoid(res[0].float()).cpu().numpy()
        cat = torch.sigmoid(res[1].float()).cpu().numpy()
        on = torch.sigmoid(res[2].float()).cpu().numpy()
        off = torch.sigmoid(res[3].float()).cpu().numpy()
        dist = res[5].float().cpu().numpy()
        lengths = batch["frame_mask"].sum(dim=1).long().numpy()
        for j, cid in enumerate(batch["clip_ids"]):
            k = lengths[j]
            out[cid] = {"act": act[j, :k], "cat": cat[j, :k], "on": on[j, :k], "off": off[j, :k],
                        "dist": dist[j, :k]}
    pickle.dump({"outputs": out, "records": ds.records, "ckpt": args.ckpt},
                open(args.out, "wb"), protocol=4)
    print(f"cached {len(out)} clips -> {args.out}")


if __name__ == "__main__":
    main()
