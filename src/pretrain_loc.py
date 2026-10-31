"""N3: pretrain BEATs on a localisation pretext - exact edges of ESC-50 events pasted into unlabelled Vaani audio at natural levels."""

from __future__ import annotations
from paths import DATA, PRETRAINED, WORK  # noqa: F401

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from dataset import VaaniWaveDataset
from overlap_head import two_colour
from pretrain_dapt import parameter_groups, set_lrs

SR, HOP = 16000, 160


class InsertBank:
    def __init__(self, path: str):
        z = np.load(path)
        self.audio, self.offsets = z["audio"], z["offsets"]

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def get(self, i: int) -> np.ndarray:
        return self.audio[self.offsets[i]: self.offsets[i + 1]].astype(np.float32)


class LocMix(Dataset):
    """A bed crop with 0-3 pasted events; event level is drawn relative to the bed over the event span."""

    def __init__(self, beds, bank: InsertBank, esr_mean: float, esr_sd: float, fixed: bool = False):
        self.beds, self.bank, self.esr_mean, self.esr_sd, self.fixed = beds, bank, esr_mean, esr_sd, fixed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        for d in self.beds.datasets:
            d.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.beds)

    def __getitem__(self, idx: int) -> dict:
        item = self.beds[idx]
        wav, pad = item["wav"].numpy().copy(), item["padding_mask"].numpy()
        valid = int((~pad).sum())
        rng = np.random.default_rng((idx * 7919 + (0 if self.fixed else self.epoch * 104729 + random.getrandbits(20))) & 0xFFFFFFFF)
        n_ev = rng.choice(4, p=[0.15, 0.35, 0.30, 0.20])
        events = []
        for _ in range(n_ev):
            ins = self.bank.get(int(rng.integers(len(self.bank))))
            dur = float(np.clip(np.exp(rng.normal(np.log(0.5), 0.8)), 0.1, min(3.5, len(ins) / SR, valid / SR - 0.05)))
            n = int(dur * SR)
            if n < int(0.1 * SR):
                continue
            a0 = int(rng.integers(0, len(ins) - n + 1))
            seg = ins[a0: a0 + n].copy()
            fade = int(0.002 * SR)
            seg[:fade] *= np.linspace(0, 1, fade); seg[-fade:] *= np.linspace(1, 0, fade)
            pos = int(rng.integers(0, valid - n + 1))
            bed_rms = np.sqrt(np.mean(wav[pos: pos + n] ** 2) + 1e-10)
            if bed_rms < 1e-4:
                bed_rms = np.sqrt(np.mean(wav[:valid] ** 2) + 1e-10)
            esr = float(np.clip(rng.normal(self.esr_mean, self.esr_sd), -22.0, 10.0))
            wav[pos: pos + n] += seg * bed_rms * 10 ** (esr / 20) / (np.sqrt(np.mean(seg ** 2)) + 1e-8)
            events.append({"onset": pos / SR, "offset": (pos + n) / SR})
        peak = np.abs(wav).max()
        if peak > 0.99:
            wav *= 0.99 / peak
        T = len(wav) // HOP
        tgt = np.zeros((T, 4), dtype=np.float32)
        grid = np.arange(T, dtype=np.float32)
        for e in events:
            a, b = int(round(e["onset"] * SR / HOP)), int(round(e["offset"] * SR / HOP))
            tgt[a:b, 0] = 1.0
            tgt[:, 1] = np.maximum(tgt[:, 1], np.exp(-0.5 * ((grid - a) / 1.0) ** 2))
            tgt[:, 2] = np.maximum(tgt[:, 2], np.exp(-0.5 * ((grid - b) / 1.0) ** 2))
        for on, off in two_colour(events)[1]:
            tgt[int(round(on * SR / HOP)): int(round(off * SR / HOP)), 3] = 1.0
        fmask = np.zeros(T, dtype=np.float32); fmask[: valid // HOP] = 1.0
        return {"wav": torch.from_numpy(wav), "padding_mask": torch.from_numpy(pad), "target": torch.from_numpy(tgt),
                "fmask": torch.from_numpy(fmask), "events": events}


def collate(batch):
    return {"wav": torch.stack([b["wav"] for b in batch]), "padding_mask": torch.stack([b["padding_mask"] for b in batch]),
            "target": torch.stack([b["target"] for b in batch]), "fmask": torch.stack([b["fmask"] for b in batch]),
            "events": [b["events"] for b in batch]}


class LocModel(torch.nn.Module):
    def __init__(self, beats):
        super().__init__()
        self.beats = beats
        self.token_head = torch.nn.Identity()          # keeps parameter_groups' head bucket well-defined
        self.mask_embed = torch.nn.Parameter(torch.zeros(1))
        dim = beats.cfg.encoder_embed_dim
        self.rnn = torch.nn.GRU(dim, 192, batch_first=True, bidirectional=True)
        self.out = torch.nn.Linear(384, 4)

    def head_parameters(self):
        return list(self.rnn.parameters()) + list(self.out.parameters())

    def forward(self, wav, pad, n_frames):
        feats, _ = self.beats.extract_features(wav, padding_mask=pad)
        feats = F.interpolate(feats.transpose(1, 2).float(), size=n_frames, mode="linear", align_corners=False).transpose(1, 2)
        with torch.autocast("cuda", enabled=False):
            x, _ = self.rnn(feats.float())
            return self.out(x)


def decode(p: np.ndarray, thr: float = 0.5) -> list[tuple[float, float]]:
    m = np.r_[0, (p >= thr).astype(np.int8), 0]
    on, off = np.flatnonzero(np.diff(m) == 1), np.flatnonzero(np.diff(m) == -1)
    return [(a * HOP / SR, b * HOP / SR) for a, b in zip(on, off) if b - a >= 5]


def edge_hit_rate(model, loader, device) -> tuple[float, float]:
    hit = n = n_pred = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(batch["wav"].to(device), batch["padding_mask"].to(device), batch["target"].shape[1])
            act = torch.sigmoid(lg[..., 0].float()).cpu().numpy()
            for j, evs in enumerate(batch["events"]):
                pred = decode(act[j]); n_pred += len(pred); used = set()
                for e in evs:
                    tol = max(0.2 * (e["offset"] - e["onset"]), 0.05); n += 1
                    for k, (po, pf) in enumerate(pred):
                        if k not in used and abs(po - e["onset"]) <= tol and abs(pf - e["offset"]) <= tol:
                            used.add(k); hit += 1; break
    model.train()
    return hit / max(n, 1), hit / max(n_pred, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--beats-ckpt", default=f"{PRETRAINED}/BEATs_iter3_plus_AS2M.pt")
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
    sys.path.insert(0, str(Path(__file__).parent / "beats"))
    from BEATs import BEATs, BEATsConfig
    ck = torch.load(args.beats_ckpt, map_location="cpu", weights_only=False)
    beats = BEATs(BEATsConfig(ck["cfg"])); beats.load_state_dict(ck["model"])
    model = LocModel(beats).to(device)
    bank = InsertBank(args.bank)
    beds = ConcatDataset([VaaniWaveDataset(m, r, 8.0, seed=args.seed, holdout_percent=1.0) for m, r in args.manifest_root])
    held = ConcatDataset([VaaniWaveDataset(m, r, 8.0, train=False, holdout_percent=1.0, holdout=True) for m, r in args.manifest_root])
    train_ds = LocMix(beds, bank, args.esr_mean, args.esr_sd)
    held_ds = LocMix(held, bank, args.esr_mean, args.esr_sd, fixed=True)
    held_ds = torch.utils.data.Subset(held_ds, list(range(0, len(held_ds), max(1, len(held_ds) // 512)))[:512])
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True,
                        drop_last=True, collate_fn=collate, persistent_workers=True)
    held_loader = DataLoader(held_ds, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate)
    groups = parameter_groups(model, args)
    groups[0]["params"] = model.head_parameters()
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.98))
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
            torch.save({"cfg": model.beats.cfg.__dict__, "model": model.beats.state_dict()}, out / f"beats_loc_{step + 1:05d}.pt")
            torch.save({"rnn": model.rnn.state_dict(), "out": model.out.state_dict(), "step": step + 1}, out / "head_last.pt")
    print("N3_PRETRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
