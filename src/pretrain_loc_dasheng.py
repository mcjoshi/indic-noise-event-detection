"""N3 pretext on Dasheng-base: exact edges of ESC-50 events pasted into unlabelled Vaani audio; saves encoder state dicts sed_encoders loads as "dasheng"."""

from __future__ import annotations
from paths import DATA, PRETRAINED, WORK  # noqa: F401

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from dasheng import dasheng_06B, dasheng_base
from dataset import VaaniWaveDataset
from pretrain_dapt import set_lrs
from pretrain_loc import InsertBank, LocMix, collate, edge_hit_rate


class LocModelDasheng(torch.nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc
        self.rnn = torch.nn.GRU(enc.embed_dim, 192, batch_first=True, bidirectional=True)
        self.out = torch.nn.Linear(384, 4)

    def forward(self, wav, pad, n_frames):
        feats = self.enc(wav.float())
        feats = F.interpolate(feats.transpose(1, 2).float(), size=n_frames, mode="linear", align_corners=False).transpose(1, 2)
        with torch.autocast("cuda", enabled=False):
            x, _ = self.rnn(feats.float())
            return self.out(x)


def parameter_groups(model, args):
    groups = [{"params": list(model.rnn.parameters()) + list(model.out.parameters()), "lr": args.head_lr, "peak_lr": args.head_lr, "weight_decay": args.weight_decay}]
    n = len(model.enc.blocks); buckets: dict[float, list] = {}
    for name, p in model.enc.named_parameters():
        if name.startswith("blocks."):
            lr = args.encoder_lr * args.layer_decay ** (n - int(name.split(".")[1]) - 1)
        elif name.startswith(("patch_embed", "time_pos_embed", "freq_pos_embed", "init_bn")):
            lr = args.patch_lr
        else:
            lr = args.encoder_lr
        buckets.setdefault(lr, []).append(p)
    groups.extend({"params": ps, "lr": lr, "peak_lr": lr, "weight_decay": args.weight_decay} for lr, ps in buckets.items())
    return groups


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", choices=("base", "06B"), default="base")
    ap.add_argument("--bank", default=f"{DATA}/n3_esc_bank.npz")
    ap.add_argument("--manifest-root", nargs=2, action="append", metavar=("MANIFEST", "ROOT"), required=True)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--encoder-lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=5e-4)
    ap.add_argument("--patch-lr", type=float, default=1e-5)
    ap.add_argument("--layer-decay", type=float, default=0.85)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--esr-mean", type=float, default=-7.6)
    ap.add_argument("--esr-sd", type=float, default=6.0)
    ap.add_argument("--bnd-weight", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = torch.device("cuda")
    enc = dasheng_06B() if args.size == "06B" else dasheng_base()
    model = LocModelDasheng(enc).to(device)
    bank = InsertBank(args.bank)
    beds = ConcatDataset([VaaniWaveDataset(m, r, 8.0, seed=args.seed, holdout_percent=1.0) for m, r in args.manifest_root])
    held = ConcatDataset([VaaniWaveDataset(m, r, 8.0, train=False, holdout_percent=1.0, holdout=True) for m, r in args.manifest_root])
    train_ds = LocMix(beds, bank, args.esr_mean, args.esr_sd)
    held_ds = LocMix(held, bank, args.esr_mean, args.esr_sd, fixed=True)
    held_ds = torch.utils.data.Subset(held_ds, list(range(0, len(held_ds), max(1, len(held_ds) // 512)))[:512])
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True,
                        drop_last=True, collate_fn=collate, persistent_workers=True)
    held_loader = DataLoader(held_ds, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate)
    optimizer = torch.optim.AdamW(parameter_groups(model, args), betas=(0.9, 0.98))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    hr, pr = edge_hit_rate(model, held_loader, device)
    print(f"train beds {len(beds)} | held-out mixtures {len(held_ds)} | inserts {len(bank)} | untrained edge hit {hr:.3f} precision {pr:.3f}", flush=True)
    it, epoch, w = iter(loader), 0, torch.tensor([1.0, args.bnd_weight, args.bnd_weight, 1.0], device=device)
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(args.grad_accum):
            try:
                batch = next(it)
            except StopIteration:
                epoch += 1; train_ds.set_epoch(epoch); it = iter(loader); batch = next(it)
            tgt, fm = batch["target"].to(device), batch["fmask"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(batch["wav"].to(device), batch["padding_mask"].to(device), tgt.shape[1])
            lo = F.binary_cross_entropy_with_logits(lg.float(), tgt, reduction="none") * w
            loss = (lo * fm[..., None]).sum() / (fm.sum() * 4) / args.grad_accum
            loss.backward(); tot += loss.item()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); set_lrs(optimizer, step, args.warmup_steps, args.steps)
        if step % 200 == 0:
            print(f"step {step + 1}/{args.steps} loss {tot:.4f} grad {gn:.3f}", flush=True)
        if (step + 1) % 1000 == 0 or step + 1 == args.steps:
            hr, pr = edge_hit_rate(model, held_loader, device)
            print(f"  held-out pretext: edge hit {hr:.3f} (both edges in collar), precision {pr:.3f}", flush=True)
            torch.save(model.enc.state_dict(), out / f"dasheng_loc_{step + 1:05d}.pt")
    print("N3DASHENG_PRETRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
