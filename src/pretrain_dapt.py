"""Continue-pretrain BEATs with frozen acoustic-token targets on Vaani audio."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import ConcatDataset, DataLoader, Sampler

from dataset import VaaniWaveDataset


def setup_dist() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 1, 0
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    return rank, world, local


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def log(message: str) -> None:
    if is_main():
        print(message, flush=True)


def make_span_mask(valid_lengths: torch.Tensor, ratio: float, span: int,
                   generator: torch.Generator) -> torch.Tensor:
    """Create nonoverlapping contiguous masks without ever selecting padding.

    Spans sit on a random offset of a `span`-aligned grid, so they never overlap by
    construction and the row is built in O(T); the budget is met exactly by trimming the
    last span."""
    if not 0 < ratio <= 1 or span < 1:
        raise ValueError("ratio must be in (0, 1] and span must be positive")
    mask = torch.zeros((len(valid_lengths), int(valid_lengths.max().item())),
                       dtype=torch.bool, device=valid_lengths.device)
    for row, length_t in enumerate(valid_lengths):
        length = int(length_t.item())
        budget = max(1, round(length * ratio))
        if length >= 2:
            budget = max(2, budget)
        budget = min(length, budget)
        width = min(span, length)
        dev = generator.device
        offset = int(torch.randint(width, (), generator=generator, device=dev).item()) if width > 1 else 0
        starts = torch.arange(offset, length, width)
        order = torch.randperm(len(starts), generator=generator, device=dev).cpu()
        remaining = budget
        for start in starts[order].tolist():
            if remaining <= 0:
                break
            w = min(width, length - start, remaining)
            mask[row, start:start + w] = True
            remaining -= w
        if remaining > 0:                      # grid could not host the budget: fill from the front
            free = (~mask[row, :length]).nonzero().flatten()[:remaining]
            mask[row, free] = True
    return mask


class MaskedTokenPredictor(torch.nn.Module):
    def __init__(self, beats, quant_n: int = 1024):
        super().__init__()
        self.beats = beats
        dim = beats.cfg.encoder_embed_dim
        self.mask_embed = torch.nn.Parameter(torch.empty(dim))
        torch.nn.init.normal_(self.mask_embed, std=0.02)
        self.token_head = torch.nn.Linear(dim, quant_n)

    def forward(self, wav, padding_mask, token_mask):
        states, token_padding = self.beats.extract_features(
            wav, padding_mask=padding_mask, token_mask=token_mask, mask_embed=self.mask_embed
        )
        return self.token_head(states), token_padding, states


class WeightedDistributedSampler(Sampler[int]):
    """Sample global weighted indices, then shard them identically across ranks."""

    def __init__(self, weights: torch.Tensor, world: int, rank: int, seed: int):
        self.weights, self.world, self.rank, self.seed = weights.double(), world, rank, seed
        self.num_samples = math.ceil(len(weights) / world)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(self.weights, self.num_samples * self.world,
                                    replacement=True, generator=generator)
        return iter(indices[self.rank::self.world].tolist())

    def __len__(self) -> int:
        return self.num_samples


def collate_wave(batch: list[dict]) -> dict:
    return {
        "clip_ids": [item["clip_id"] for item in batch],
        "wav": torch.stack([item["wav"] for item in batch]),
        "padding_mask": torch.stack([item["padding_mask"] for item in batch]),
    }


def parameter_groups(model: MaskedTokenPredictor, args):
    groups = [{"params": list(model.token_head.parameters()) + [model.mask_embed],
               "lr": args.head_lr, "peak_lr": args.head_lr, "weight_decay": args.weight_decay}]
    n_layers = len(model.beats.encoder.layers)
    buckets: dict[float, list] = {}
    for name, parameter in model.beats.named_parameters():
        if name.startswith("encoder.layers."):
            depth = int(name.split(".")[2]) + 1
            lr = args.encoder_lr * args.layer_decay ** (n_layers - depth)
        elif name.startswith(("patch_embedding", "post_extract_proj")):
            lr = args.patch_lr
        else:
            lr = args.encoder_lr
        buckets.setdefault(lr, []).append(parameter)
    groups.extend({"params": params, "lr": lr, "peak_lr": lr, "weight_decay": args.weight_decay}
                  for lr, params in buckets.items())
    return groups


def set_lrs(optimizer, step, warmup, total):
    if step < warmup:
        scale = (step + 1) / max(1, warmup)
    else:
        progress = (step - warmup) / max(1, total - warmup)
        scale = 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1, progress)))
    for group in optimizer.param_groups:
        group["lr"] = group["peak_lr"] * scale


def reduce_scalars(values: dict[str, torch.Tensor]) -> dict[str, float]:
    out = {}
    for key, value in values.items():
        value = value.detach().float().clone()
        if dist.is_initialized():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            value /= dist.get_world_size()
        out[key] = value.item()
    return out


def checkpoint(path, model, optimizer, step, args, tokenizer_sha):
    raw = model.module if isinstance(model, DistributedDataParallel) else model
    torch.save({"student": raw.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                "scheduler": {"step": step}, "args": vars(args),
                "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                "python": random.getstate()}, "tokenizer_sha256": tokenizer_sha}, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--beats-ckpt", required=True)
    ap.add_argument("--tokenizer-ckpt", required=True)
    ap.add_argument("--manifest-root", nargs=2, action="append", metavar=("MANIFEST", "ROOT"), required=True)
    ap.add_argument("--steps", type=int, default=25000)
    ap.add_argument("--crop-sec", type=float, default=8.0)
    ap.add_argument("--mask-ratio", type=float, default=0.65)
    ap.add_argument("--mask-span", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--encoder-lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=2e-4)
    ap.add_argument("--patch-lr", type=float, default=1e-5)
    ap.add_argument("--layer-decay", type=float, default=0.85)
    ap.add_argument("--warmup-steps", type=int, default=2000)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout-percent", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume")
    args = ap.parse_args()
    rank, world, local = setup_dist()
    device = torch.device(f"cuda:{local}")
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)

    import sys
    sys.path.insert(0, str(Path(__file__).parent / "beats"))
    from BEATs import BEATs, BEATsConfig
    from Tokenizers import Tokenizers, TokenizersConfig
    encoder_ck = torch.load(args.beats_ckpt, map_location="cpu", weights_only=False)
    tokenizer_ck = torch.load(args.tokenizer_ckpt, map_location="cpu", weights_only=False)
    tokenizer = Tokenizers(TokenizersConfig(tokenizer_ck["cfg"])).to(device)
    tokenizer.load_state_dict(tokenizer_ck["model"])
    tokenizer.eval()
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)
    student = MaskedTokenPredictor(BEATs(BEATsConfig(encoder_ck["cfg"])), tokenizer.quant_n)
    student.beats.load_state_dict(encoder_ck["model"])
    if student.token_head.out_features != tokenizer.quant_n:
        raise RuntimeError("student vocabulary does not match tokenizer.quant_n")
    student = student.to(device)

    datasets = [VaaniWaveDataset(manifest, root, args.crop_sec, seed=args.seed,
                                 holdout_percent=args.holdout_percent)
                for manifest, root in args.manifest_root]
    if len(datasets) < 2:
        raise ValueError("supply parent manifests plus the labelled manifest")
    lengths = [len(ds) for ds in datasets]
    weights = torch.cat([torch.full((n,), 0.9 / sum(lengths[:-1])) for n in lengths[:-1]]
                        + [torch.full((lengths[-1],), 0.1 / lengths[-1])])
    data = ConcatDataset(datasets)
    sampler = WeightedDistributedSampler(weights, world, rank, args.seed)
    loader = DataLoader(data, batch_size=args.batch_size, sampler=sampler, shuffle=False,
                        num_workers=args.workers, pin_memory=True, drop_last=True,
                        collate_fn=collate_wave)
    optimizer = torch.optim.AdamW(parameter_groups(student, args), betas=(0.9, 0.98))
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        student.load_state_dict(resume["student"])
        optimizer.load_state_dict(resume["optimizer"])
        start_step = resume["step"]
    else:
        start_step = 0
    # layerdrop skips whole blocks on some steps, so DDP must tolerate unused parameters
    student = (DistributedDataParallel(student, device_ids=[local], find_unused_parameters=True)
               if world > 1 else student)
    out = Path(args.out)
    if is_main():
        out.mkdir(parents=True, exist_ok=True)
        (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    tokenizer_sha = hashlib.sha256(Path(args.tokenizer_ckpt).read_bytes()).hexdigest()
    holdout_count = sum(len(VaaniWaveDataset(manifest, root, args.crop_sec, train=False,
                                              holdout_percent=args.holdout_percent, holdout=True))
                        for manifest, root in args.manifest_root[:-1])
    monitor_data = ConcatDataset([
        VaaniWaveDataset(manifest, root, args.crop_sec, train=False,
                         holdout_percent=args.holdout_percent, holdout=True)
        for manifest, root in args.manifest_root[:-1]
    ])
    monitor_loader = DataLoader(monitor_data, batch_size=args.batch_size, shuffle=False,
                                num_workers=0, collate_fn=collate_wave)
    monitor_iterator = iter(monitor_loader)
    log(f"train {sum(lengths)} clips | monitor holdout {holdout_count} clips | world {world} | "
        f"tokenizer codes {tokenizer.quant_n}")
    iterator, epoch, collapse = iter(loader), 0, 0
    for step in range(start_step, args.steps):
        sampler.set_epoch(epoch)
        for dataset in datasets:
            dataset.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        totals = {}
        for accumulation in range(args.grad_accum):
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                for dataset in datasets:
                    dataset.set_epoch(epoch)
                iterator = iter(loader)
                batch = next(iterator)
            wav, sample_padding = batch["wav"].to(device), batch["padding_mask"].to(device)
            with torch.no_grad():
                labels = tokenizer.extract_labels(wav, sample_padding)
                labels = labels.view(wav.shape[0], -1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                patch, token_padding = (student.module if isinstance(student, DistributedDataParallel) else student).beats.extract_patch_features(wav, sample_padding)
                if labels.shape[1] != patch.shape[1]:
                    raise RuntimeError(f"token length mismatch: labels {labels.shape[1]}, student {patch.shape[1]}")
                valid = (~token_padding).sum(dim=1)
                generator = torch.Generator(device=device).manual_seed(args.seed + step * 1009 + rank * 9176 + accumulation)
                token_mask = make_span_mask(valid, args.mask_ratio, args.mask_span, generator)
                token_mask = F.pad(token_mask, (0, patch.shape[1] - token_mask.shape[1]))
                logits, _, states = student(wav, sample_padding, token_mask)
                loss = F.cross_entropy(logits[token_mask].float(), labels[token_mask].long()) / args.grad_accum
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite masked-token loss")
            loss.backward()
            totals["loss"] = totals.get("loss", 0) + loss.detach()
            totals["acc"] = totals.get("acc", 0) + (logits[token_mask].argmax(-1) == labels[token_mask]).float().mean() / args.grad_accum
            totals["state_std"] = totals.get("state_std", 0) + states.float().std() / args.grad_accum
            totals["state_norm"] = totals.get("state_norm", 0) + states.float().norm(dim=-1).mean() / args.grad_accum
            target_codes = torch.bincount(labels[token_mask], minlength=tokenizer.quant_n).float()
            pred_codes = torch.bincount(logits[token_mask].argmax(-1), minlength=tokenizer.quant_n).float()
            if dist.is_initialized():
                dist.all_reduce(target_codes)
                dist.all_reduce(pred_codes)
            for prefix, codes in (("target", target_codes), ("pred", pred_codes)):
                probs = codes / codes.sum().clamp_min(1)
                perplexity = torch.exp(-(probs[probs > 0] * probs[probs > 0].log()).sum())
                totals[f"{prefix}_ppl"] = totals.get(f"{prefix}_ppl", 0) + perplexity / args.grad_accum
                if prefix == "target":
                    totals["target_used"] = totals.get("target_used", 0) + (codes > 0).sum() / args.grad_accum
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        set_lrs(optimizer, step, args.warmup_steps, args.steps)
        if step % 100 == 0 or step + 1 == args.steps:
            metrics = reduce_scalars({**totals, "grad_norm": grad_norm})
            log(f"step {step + 1}/{args.steps} loss {metrics['loss']:.4f} acc {metrics['acc']:.4f} "
                f"target_codes {metrics['target_used']:.0f} target_ppl {metrics['target_ppl']:.1f} "
                f"pred_ppl {metrics['pred_ppl']:.1f} state_norm {metrics['state_norm']:.3f} "
                f"state_std {metrics['state_std']:.5f} grad {metrics['grad_norm']:.3f}")
            if metrics["state_std"] < 1e-3:
                raise RuntimeError("encoder-state standard deviation collapsed")
            # the fresh head predicts the marginal code for the first few hundred steps; judge collapse only after warm-up
            armed = step + 1 > args.warmup_steps + 1000
            collapse = collapse + 100 if armed and metrics["pred_ppl"] < 0.05 * metrics["target_ppl"] else 0
            if collapse >= 500:
                raise RuntimeError("prediction-code perplexity collapsed for 500 steps")
        if (step + 1) % 100 == 0:
            try:
                monitor_batch = next(monitor_iterator)
            except StopIteration:
                monitor_iterator = iter(monitor_loader)
                monitor_batch = next(monitor_iterator)
            student.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                monitor_wav = monitor_batch["wav"].to(device)
                monitor_padding = monitor_batch["padding_mask"].to(device)
                monitor_labels = tokenizer.extract_labels(monitor_wav, monitor_padding)
                monitor_labels = monitor_labels.view(monitor_wav.shape[0], -1)
                monitor_patch, monitor_token_padding = (
                    student.module if isinstance(student, DistributedDataParallel) else student
                ).beats.extract_patch_features(monitor_wav, monitor_padding)
                monitor_mask = make_span_mask(
                    (~monitor_token_padding).sum(1), args.mask_ratio, args.mask_span,
                    torch.Generator(device=device).manual_seed(args.seed + step + rank),
                )
                monitor_mask = F.pad(monitor_mask, (0, monitor_patch.shape[1] - monitor_mask.shape[1]))
                monitor_logits, _, _ = student(monitor_wav, monitor_padding, monitor_mask)
                monitor_loss = F.cross_entropy(
                    monitor_logits[monitor_mask].float(), monitor_labels[monitor_mask].long()
                )
            student.train()
            monitor = reduce_scalars({"holdout_loss": monitor_loss})
            log(f"  holdout masked-token loss {monitor['holdout_loss']:.4f}")
        if (step + 1) % 1000 == 0 or step + 1 == args.steps:
            if dist.is_initialized():
                dist.barrier()
            if is_main():
                checkpoint(out / f"resume_{step + 1:05d}.pt", student, optimizer, step + 1, args, tokenizer_sha)
        if (step + 1) % 5000 == 0 or step + 1 == args.steps:
            if is_main():
                raw = student.module if isinstance(student, DistributedDataParallel) else student
                torch.save({"cfg": raw.beats.cfg.__dict__, "model": raw.beats.state_dict()},
                           out / f"beats_dapt_{step + 1:05d}.pt")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
