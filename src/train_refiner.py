"""Train the boundary refiner on Gold training clips, evaluate it on validation.

The refiner is fitted on *training* clips and judged on `val_mixed`, so its benefit is
measured on data neither it nor the decode config was fitted to.  Fitting it on val would
make the reported gain meaningless, since val is also where the decode parameters are
selected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from boundary_refiner import (BoundaryRefiner, build_decoder_examples,
                              build_examples, refine_spans)
from dataset import VaaniFrameDataset, collate
from ensemble import load_cache
from model import load_checkpoint
from scorer import score
from tune_postproc import decode_one


@torch.no_grad()
def cache_curves(ckpt, manifest, audio_root, device, batch_size=32, workers=10, limit=0):
    model, _ = load_checkpoint(ckpt, device, use_ema=True)
    ds = VaaniFrameDataset(manifest, audio_root, max_sec=None, train=False)
    if limit:
        ds.records = ds.records[:limit]
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                    collate_fn=collate, pin_memory=True)
    by_id = {r["clip_id"]: r for r in ds.records}
    out = []
    for bi, batch in enumerate(dl):
        wav = batch["wav"].to(device, non_blocking=True)
        n_fr = int(batch["frame_mask"].shape[1])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            res = model(wav, n_fr)
        act = torch.sigmoid(res[0].float()).cpu().numpy()
        on = torch.sigmoid(res[2].float()).cpu().numpy()
        off = torch.sigmoid(res[3].float()).cpu().numpy()
        lengths = batch["frame_mask"].sum(dim=1).long().numpy()
        for j, cid in enumerate(batch["clip_ids"]):
            k = lengths[j]
            rec = by_id[cid]
            out.append({"clip_id": cid, "activity": act[j, :k], "onset": on[j, :k],
                        "offset": off[j, :k], "duration": rec["duration"],
                        "events": rec["events"]})
        if bi % 50 == 0:
            print(f"  {min((bi+1)*batch_size, len(ds.records))}/{len(ds.records)}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train-manifest", required=True)
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--val-cache", required=True, help="val_mixed probability cache")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=12000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=128)
    args = ap.parse_args()

    device = "cuda"
    rng = np.random.default_rng(0)

    print("caching curves on training clips...")
    clips = cache_curves(args.ckpt, args.train_manifest, args.audio_root, device,
                         limit=args.limit)
    cfg_train = json.loads(Path(args.config).read_text())["config"]
    X, K, Y = build_examples(clips, rng)
    print(f"{len(X)} jittered proposals from {len(clips)} clips")

    # plus the decoder's real proposals, whose errors are asymmetric in a way uniform
    # jitter cannot reproduce
    Xd, Kd, Yd = build_decoder_examples(
        clips,
        lambda c: decode_one(c["activity"], c["duration"], cfg_train,
                             (c["onset"], c["offset"])),
    )
    print(f"{len(Xd)} decoder-derived proposals")
    if len(Xd):
        X = np.concatenate([X, Xd]); K = np.concatenate([K, Kd]); Y = np.concatenate([Y, Yd])
    print(f"{len(X)} total training proposals")

    model = BoundaryRefiner(hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    Xt = torch.from_numpy(X).to(device)
    Kt = torch.from_numpy(K).to(device)
    Yt = torch.from_numpy(Y).to(device)
    n = len(Xt)
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i : i + args.batch]
            logp = torch.log_softmax(model(Xt[idx], Kt[idx]), dim=-1)
            loss = -(Yt[idx] * logp).sum(-1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            tot += loss.item() * len(idx)
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d} loss {tot/n:.4f}", flush=True)
    model.eval()

    # evaluate on val: decode as usual, then refine, and compare
    probs, durations, records, bnd = load_cache(args.val_cache)
    cfg = json.loads(Path(args.config).read_text())["config"]
    refs = {r["clip_id"]: [(e["onset"], e["offset"]) for e in r["events"]] for r in records}
    syn = {r["clip_id"] for r in records if r.get("synthetic")}

    base, refined = {}, {}
    for cid, p in probs.items():
        spans = decode_one(p, durations[cid], cfg, bnd.get(cid))
        base[cid] = spans
        on, off = bnd[cid]
        refined[cid] = refine_spans(spans, p, on, off, model, device, durations[cid])

    print()
    for tag, preds in (("decode only", base), ("+ refiner", refined)):
        for gname, ids in (("all", set(refs)), ("natural", set(refs) - syn), ("synthetic", syn)):
            r = {k: v for k, v in refs.items() if k in ids}
            d = {k: v for k, v in durations.items() if k in ids}
            p = {k: v for k, v in preds.items() if k in ids}
            s = score(r, p, d)
            print(f"  {tag:<12} {gname:<10} Combined {s.combined:.4f}  "
                  f"F1 {s.event_f1:.4f}  Dice {s.segment_dice:.4f}")
        print()

    torch.save({"model": model.state_dict(), "hidden": args.hidden}, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
