"""Train the CRNN noise event detector on all three annotation tiers.

Loss per tier, following the survey's recommended joint reliability-weighted scheme:
  Gold    - full-weight frame loss on soft targets
  Silver  - frame loss at reduced weight, boundaries blurred, background down-weighted
  Bronze  - clip-level MIL loss only (linear-softmax pooling); never a frame target

Single GPU:
    python3 train.py --train ... --val ... --audio-root ... --out work/run1
Multi GPU:
    torchrun --nproc_per_node=2 train.py --train ... (same flags)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from dataset import FRAME_SEC, HOP, VaaniFrameDataset, collate
from epn_head import collar_loss, collar_loss_v2, epn_decode, epn_loss
from model import CRNN, CRNNAtst, CRNNBeats, CRNNBeatsSpeech, CRNNSedEnc, decode, linear_softmax_pool, snap_to_boundaries
from scorer import score


def setup_dist() -> tuple[int, int, int]:
    frac = float(os.environ.get("VAANI_MEM_FRAC", "0"))  # memory cap when sharing a GPU
    if frac > 0:
        torch.cuda.set_per_process_memory_fraction(frac)
    if "RANK" not in os.environ:
        return 0, 1, 0
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    return rank, world, local


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def log(msg: str) -> None:
    if is_main():
        print(msg, flush=True)


def consistency_view(wav: torch.Tensor, res: torch.Tensor | None, args):
    """Student view for the consistency loss: frame k of the view is frame k+shift of the clean clip."""
    bs = wav.shape[0]
    shift = int(np.random.randint(0, args.consistency_shift + 1))
    if shift > 0:
        wav = F.pad(wav[:, shift * HOP:], (0, shift * HOP))
        if res is not None:
            res = F.pad(res[:, shift * HOP:], (0, shift * HOP))
    gain = torch.from_numpy((10 ** (np.random.uniform(-args.consistency_gain, args.consistency_gain, bs) / 20)).astype(np.float32)).to(wav.device).view(-1, 1)
    snr = torch.from_numpy(np.random.uniform(args.consistency_snr - 10, args.consistency_snr + 10, bs).astype(np.float32)).to(wav.device).view(-1, 1)
    wav = wav * gain
    rms = wav.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-5)
    wav = wav + torch.randn_like(wav) * rms / 10 ** (snr / 20)
    if res is not None:
        res = res * gain
    return wav, res, shift


def mixup(batch: dict, alpha: float, prob: float, generator: torch.Generator | None = None):
    """Waveform mixup with union targets.

    Mixing two clips means events from both are genuinely audible, so targets combine
    with max rather than a convex blend.  Pairs are only mixed when both sides agree on
    whether frame labels exist, so a Bronze clip can never contaminate a frame target.
    """
    bs = batch["wav"].shape[0]
    perm = torch.randperm(bs, device=batch["wav"].device, generator=generator)
    same_kind = batch["has_frames"] == batch["has_frames"][perm]
    do_mix = (torch.rand(bs, device=batch["wav"].device) < prob) & same_kind
    if not do_mix.any():
        return batch

    lam = torch.from_numpy(
        np.random.beta(alpha, alpha, size=bs).astype(np.float32)
    ).to(batch["wav"].device)
    lam = torch.where(do_mix, lam, torch.ones_like(lam))

    batch["wav"] = lam[:, None] * batch["wav"] + (1 - lam[:, None]) * batch["wav"][perm]
    if batch.get("res") is not None:
        batch["res"] = lam[:, None] * batch["res"] + (1 - lam[:, None]) * batch["res"][perm]
    m = do_mix[:, None].float()
    # Mixing two clips makes both sources audible at once, so any frame active in BOTH
    # is genuinely two simultaneous events and belongs to layer 2.  This has to be
    # computed from the pre-mix targets, and it makes mixup an overlap augmentation -
    # useful because only ~15% of real clips contain an overlap to learn from.
    new_l2 = batch["target"] * batch["target"][perm] * m
    # a partner event makes the primary clip's proposal targets ambiguous on those frames
    partner_active = (batch["target"][perm] > 0.5).float() * m
    batch["mask_epn"] = batch["mask_epn"] * (1.0 - partner_active)[:, :, None]
    batch["target_l2"] = torch.maximum(
        torch.maximum(batch["target_l2"], batch["target_l2"][perm] * m), new_l2)
    batch["target"] = torch.maximum(batch["target"], batch["target"][perm] * m)
    batch["target_cat"] = torch.maximum(batch["target_cat"], batch["target_cat"][perm] * m[:, :, None])
    # the boundary targets must be combined too, or a mixed clip is supervised with the
    # boundaries of only one of its two sources
    batch["event_weight"] = torch.maximum(batch["event_weight"], batch["event_weight"][perm] * m)
    batch["target_onset"] = torch.maximum(batch["target_onset"], batch["target_onset"][perm] * m)
    batch["target_offset"] = torch.maximum(batch["target_offset"], batch["target_offset"][perm] * m)
    batch["frame_mask"] = torch.maximum(batch["frame_mask"], batch["frame_mask"][perm] * m)
    batch["clip_cat"] = torch.maximum(batch["clip_cat"], batch["clip_cat"][perm] * m)
    batch["collar_mask"] = batch["collar_mask"] * (1.0 - do_mix.float())
    return batch


def frame_loss(logits, target, mask, weight, bg_weight, event_w=None):
    """BCE over valid frames, per-clip weighted, with background frames down-weighted.

    Down-weighting background follows the finding that halving the background loss
    contribution improved macro/micro F1 by ~9% when timestamps are noisy: an unverified
    Silver boundary is far more likely to mislabel silence as event, or the reverse, in
    the background region than in the event core.
    """
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    fw = target + (1.0 - target) * bg_weight.view(-1, 1)
    if event_w is not None:
        fw = fw * event_w          # equalise each event's contribution regardless of length
    w = mask * fw * weight.view(-1, 1)
    return (loss * w).sum() / w.sum().clamp(min=1.0)


@torch.no_grad()
def run_inference(model, loader, device, decode_kwargs, max_batches=None, epn_cfg=None):
    model.eval()
    preds, raw, preds_epn = {}, {}, {}
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        wav = batch["wav"].to(device, non_blocking=True)
        n_fr = int(batch["frame_mask"].shape[1])
        res = batch["res"].to(device, non_blocking=True) if batch.get("res") is not None else None
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, _, log_on, log_off, _, epn = model(wav, n_fr, res)
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        p_on = torch.sigmoid(log_on.float()).cpu().numpy()
        p_off = torch.sigmoid(log_off.float()).cpu().numpy()
        dist = epn.float().cpu().numpy()
        lengths = batch["frame_mask"].sum(dim=1).long().numpy()
        for j, clip_id in enumerate(batch["clip_ids"]):
            k = lengths[j]
            p = probs[j, :k]
            raw[clip_id] = p.astype(np.float16)
            spans = decode(p, duration=batch["durations"][j], **decode_kwargs)
            preds[clip_id] = snap_to_boundaries(spans, p_on[j, :k], p_off[j, :k])
            if epn_cfg is not None:
                preds_epn[clip_id] = epn_decode(p, dist[j, :k, 0], dist[j, :k, 1],
                                                duration=batch["durations"][j], **epn_cfg)
    model.train()
    if epn_cfg is not None:
        return preds, raw, preds_epn
    return preds, raw


def evaluate(ema, val_dl, device, track_decode, track_epn, refs, durations, tag):
    """Score the EMA model; with an EPN in training, checkpoint selection follows the
    proposal decode because that is the system that ships."""
    if track_epn is None:
        preds, _ = run_inference(ema, val_dl, device, track_decode)
        s = score(refs, preds, durations)
        log(f"  [eval @ {tag}] EMA F1 {s.event_f1:.4f} Dice {s.segment_dice:.4f} "
            f"Combined {s.combined:.4f}")
        return s
    preds, _, preds_epn = run_inference(ema, val_dl, device, track_decode, epn_cfg=track_epn)
    s = score(refs, preds, durations)
    se = score(refs, preds_epn, durations)
    log(f"  [eval @ {tag}] EMA threshold-decode F1 {s.event_f1:.4f} Dice {s.segment_dice:.4f} "
        f"Combined {s.combined:.4f} | EPN-decode F1 {se.event_f1:.4f} Dice "
        f"{se.segment_dice:.4f} Combined {se.combined:.4f}")
    return se


def short_run_mask(target: torch.Tensor, max_len: int) -> torch.Tensor:
    """Frames inside positive runs no longer than max_len: the sub-second events that hold 59% of the Dice deficit."""
    t = (target > 0.5).cpu().numpy()
    out = np.zeros(t.shape, dtype=np.float32)
    for i, row in enumerate(t):
        d = np.diff(np.r_[0, row.astype(np.int8), 0])
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            if b - a <= max_len:
                out[i, a:b] = 1.0
    return torch.from_numpy(out).to(target.device)


def long_run_mask(target: torch.Tensor, min_len: int) -> torch.Tensor:
    t = (target > 0.5).cpu().numpy()
    out = np.zeros(t.shape, dtype=np.float32)
    for i, row in enumerate(t):
        d = np.diff(np.r_[0, row.astype(np.int8), 0])
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            if b - a >= min_len:
                out[i, a:b] = 1.0
    return torch.from_numpy(out).to(target.device)


def edge_distances(target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per frame: distance in frames to the nearest positive frame (0 inside) and to the nearest negative frame (0 outside)."""
    pos = target > 0.5
    idx = torch.arange(target.shape[1], device=target.device, dtype=torch.float32).expand_as(target)
    big = 1e6

    def nearest(mask: torch.Tensor) -> torch.Tensor:
        last = torch.where(mask, idx, torch.full_like(idx, -big)).cummax(1).values
        nxt = torch.where(mask, idx, torch.full_like(idx, big)).flip(1).cummin(1).values.flip(1)
        return torch.minimum(idx - last, nxt - idx)

    return pos, nearest(pos), nearest(~pos)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--max-sec", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--aux-weight", type=float, default=0.5)
    ap.add_argument("--clip-weight", type=float, default=0.5)
    ap.add_argument("--bnd-weight", type=float, default=1.0)
    ap.add_argument("--l2-gold-only", action="store_true",
                    help="supervise the layer-2 channel on Gold only; pseudo-labels carry no overlaps")
    ap.add_argument("--l2-weight", type=float, default=1.0,
                    help="weight on the overlapping-event channel")
    ap.add_argument("--gold-sigma", type=float, default=2.0)
    ap.add_argument("--silver-sigma", type=float, default=7.0)
    ap.add_argument("--silver-weight", type=float, default=0.5)
    ap.add_argument("--silver-blur", type=float, default=1.5)
    ap.add_argument("--silver-bg-weight", type=float, default=0.5)
    ap.add_argument("--bronze-pseudo-weight", type=float, default=0.35)
    ap.add_argument("--parent-pseudo-weight", type=float, default=0.5)
    ap.add_argument("--synth-weight", type=float, default=1.0)
    ap.add_argument("--parent-bnd-weight", type=float, default=1.0,
                    help="boundary and EPN loss multiplier for parent_pseudo clips")
    ap.add_argument("--gold-oversample", type=int, default=1,
                    help="repeat natural Gold records this many times in the training index")
    ap.add_argument("--bnd-weight-strength", type=float, default=0.0,
                    help="up-weight frames near boundaries in the primary loss")
    ap.add_argument("--event-norm", action="store_true",
                    help="weight frames so every event contributes equally")
    ap.add_argument("--mixup-alpha", type=float, default=0.2)
    ap.add_argument("--mixup-prob", type=float, default=0.5)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--conv-heads", action="store_true",
                    help="second activity/boundary heads for the Silver/pseudo convention; Gold trains and inference uses the main heads")
    ap.add_argument("--bdist-k", type=int, default=0,
                    help="distributional boundary head over +-K frames (0 = off); trained with distribution focal loss on Gold edges")
    ap.add_argument("--bdist-weight", type=float, default=1.0)
    ap.add_argument("--speed-aug", default="", help="comma-separated tape-speed factors sampled per clip, e.g. 0.9,1.0,1.1")
    ap.add_argument("--gain-aug-db", type=float, default=0.0, help="per-clip random gain in [-g, g] dB at training time")
    ap.add_argument("--frontend", choices=("logmel", "both"), default="logmel", help="'both' adds a trainable PCEN channel of the mixture")
    ap.add_argument("--epn-bins", type=int, default=0, help="distributional EPN distances over this many 10 ms bins (0 = off)")
    ap.add_argument("--epn-dfl-weight", type=float, default=1.0, help="distribution focal loss weight on Gold frames for --epn-bins")
    ap.add_argument("--init-from", default=None,
                    help="checkpoint to warm-start from, for a Gold-only refinement stage")
    ap.add_argument("--channels", default="32,64,128,128,128,128,128")
    ap.add_argument("--rnn-hidden", type=int, default=128)
    ap.add_argument("--beats-ckpt", default=None,
                    help="AudioSet-pretrained BEATs checkpoint; enables the fusion branch")
    ap.add_argument("--beats-dim", type=int, default=256)
    ap.add_argument("--atst-ckpt", default=None,
                    help="official ATST-Frame small checkpoint; enables the fusion branch")
    ap.add_argument("--atst-dim", type=int, default=256)
    ap.add_argument("--unfreeze-atst", action="store_true")
    ap.add_argument("--sed-encoder", choices=("fpasst", "m2d", "asit", "atstf", "wavlm", "wavlm_large", "xlsr", "hubert_large", "dasheng", "dasheng06b", "whisper_medium", "whisper_large"), default=None,
                    help="AudioSet-Strong PretrainedSED backbone for fusion")
    ap.add_argument("--sed-ckpt", default=None,
                    help="PretrainedSED strong checkpoint; enables the fusion branch")
    ap.add_argument("--sed-dim", type=int, default=256)
    ap.add_argument("--unfreeze-sed", action="store_true")
    ap.add_argument("--fms-prob", type=float, default=0.0,
                    help="frequency-wise MixStyle probability (DCASE device-mismatch remedy)")
    ap.add_argument("--fms-alpha", type=float, default=0.3)
    ap.add_argument("--eq-prob", type=float, default=0.0)
    ap.add_argument("--eq-max-db", type=float, default=6.0)
    ap.add_argument("--unfreeze-beats", action="store_true")
    ap.add_argument("--beats-lr-scale", type=float, default=0.1)
    ap.add_argument("--beats-layer-decay", type=float, default=1.0,
                    help="<1 lowers the lr of deeper (earlier) BEATs blocks geometrically")
    ap.add_argument("--speech-ckpt", default=None,
                    help="frozen Vaani Whisper encoder as a third branch (HF id or path)")
    ap.add_argument("--speech-dim", type=int, default=128)
    ap.add_argument("--residual-root", default=None,
                    help="directory of enhancement residuals mirroring --audio-root: adds a noise-view input channel")
    ap.add_argument("--occ-hidden", type=int, default=0, help="hidden size of the 100 ms occupancy branch (0 = off)")
    ap.add_argument("--soft-weight", type=float, default=0.0, help="mix of the ensemble vote-fraction frame target (row field soft) into the activity target")
    ap.add_argument("--long-weight", type=float, default=0.0, help="extra activity BCE on Gold frames inside long runs")
    ap.add_argument("--long-min-sec", type=float, default=4.0)
    ap.add_argument("--long-all-tiers", action="store_true", help="apply --long-weight to every framed tier, not Gold only")
    ap.add_argument("--smooth-weight", type=float, default=0.0, help="truncated-MSE smoothing of log-probabilities between neighbouring frames (MS-TCN), masked within 5 frames of a label edge")
    ap.add_argument("--smooth-tau", type=float, default=1.0)
    ap.add_argument("--long-bnd-weight", type=float, default=0.0, help="extra onset/offset BCE on Gold frames within 0.2 s of a long run's edges")
    ap.add_argument("--pu-near-sec", type=float, default=0.0,
                    help="non-Gold negatives within this many seconds of a labelled event get --pu-near-weight (0 = off)")
    ap.add_argument("--pu-near-weight", type=float, default=0.1)
    ap.add_argument("--dice-weight", type=float, default=0.0,
                    help="soft-Dice term on the activity head (Segment-Dice is half the metric)")
    ap.add_argument("--short-weight", type=float, default=0.0,
                    help="extra BCE weight on Gold frames inside runs up to --short-max-sec; the mirror of --long-weight")
    ap.add_argument("--short-max-sec", type=float, default=1.0)
    ap.add_argument("--dice-thr", type=float, default=0.0,
                    help="centre the soft-Dice sigmoid on the DECODE threshold instead of 0.5, so the loss sees the mask the scorer scores (0 = off)")
    ap.add_argument("--dice-tau", type=float, default=0.1,
                    help="temperature of that sigmoid; smaller is a sharper, more metric-like surrogate")
    ap.add_argument("--dice-medk", type=int, default=0,
                    help="smooth the probability curve over this many frames before the Dice term, mirroring the decoder's median filter (0 = off)")
    ap.add_argument("--dice-smooth", type=float, default=1.0,
                    help="soft-Dice smoothing constant; 1.0 is 2-3 percent of the denominator on a short-event clip and damps its gradient")
    ap.add_argument("--dice-alpha", type=float, default=0.5,
                    help="Tversky false-positive weight for --dice-weight; 0.5 is plain Dice, above 0.5 punishes over-prediction")
    ap.add_argument("--bdl-weight", type=float, default=0.0,
                    help="boundary loss: activity times signed distance to the nearest reference edge, so far-out frames cost most (0 = off)")
    ap.add_argument("--bdl-clip", type=int, default=50, help="distance cap in frames for --bdl-weight")
    ap.add_argument("--edge-margin-weight", type=float, default=0.0,
                    help="hinge pushing the activity logit past the decode threshold by a margin near Gold edges (0 = off)")
    ap.add_argument("--edge-margin", type=float, default=1.0, help="margin in logits for --edge-margin-weight")
    ap.add_argument("--edge-band", type=int, default=5, help="frames either side of a Gold edge that --edge-margin-weight covers")
    ap.add_argument("--edge-thr", type=float, default=0.75, help="decode threshold the margin is measured from")
    ap.add_argument("--ohem-weight", type=float, default=0.0,
                    help="BCE on only the hardest --ohem-frac of each Gold clip's frames (0 = off)")
    ap.add_argument("--ohem-frac", type=float, default=0.2)
    ap.add_argument("--epn-weight", type=float, default=0.0,
                    help="weight of the event-proposal IoU loss; 0 leaves the head untrained")
    ap.add_argument("--epn-hidden", type=int, default=128)
    ap.add_argument("--collar-weight", type=float, default=0.0,
                    help="weight of Gold-only EPN soft collar-F1 loss; 0 disables it")
    ap.add_argument("--collar-rampup", type=int, default=1000,
                    help="steps to linearly ramp --collar-weight")
    ap.add_argument("--collar-v2", action="store_true",
                    help="decoder-in-the-loop soft collar-F1 with the Gaussian collar reward, sigma annealed 1.0 -> 0.35 collar over --collar-anneal steps")
    ap.add_argument("--collar-anneal", type=int, default=6000)
    ap.add_argument("--cpu", action="store_true",
                    help="run a single-process CPU training job")
    ap.add_argument("--consistency-weight", type=float, default=0.0,
                    help="mean-teacher: MSE between student and EMA-teacher frame probabilities "
                         "on every clip, the only frame-level signal Bronze ever receives")
    ap.add_argument("--consistency-rampup", type=int, default=1000)
    ap.add_argument("--consistency-aug", action="store_true",
                    help="student sees a shifted, re-gained, noised view of the clip; the teacher sees the clean one")
    ap.add_argument("--consistency-shift", type=int, default=20, help="max student time shift in frames")
    ap.add_argument("--consistency-gain", type=float, default=6.0, help="student gain range in dB")
    ap.add_argument("--consistency-snr", type=float, default=25.0, help="centre of the student's additive-noise SNR (uniform +-10 dB)")
    ap.add_argument("--consistency-epn-weight", type=float, default=0.0,
                    help="also make the proposal head's interval regressions agree with the teacher's on frames the teacher calls active")
    args = ap.parse_args()

    if args.cpu and "RANK" in os.environ:
        raise RuntimeError("--cpu does not support distributed training")
    rank, world, local = setup_dist()
    device = torch.device("cpu" if args.cpu else f"cuda:{local}")
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    out_dir = Path(args.out)
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "args.json", "w") as fh:
            json.dump(vars(args), fh, indent=2)

    train_ds = VaaniFrameDataset(
        args.train, args.audio_root, max_sec=args.max_sec, train=True,
        silver_blur=args.silver_blur, gold_sigma=args.gold_sigma,
        silver_sigma=args.silver_sigma,
        bnd_weight_strength=args.bnd_weight_strength, soft_weight=args.soft_weight, seed=args.seed + rank,
        residual_root=args.residual_root,
        gold_oversample=args.gold_oversample,
        gain_aug_db=args.gain_aug_db,
        speed_aug=args.speed_aug,
    )
    val_ds = VaaniFrameDataset(args.val, args.audio_root, max_sec=None, train=False,
                               residual_root=args.residual_root)
    log(f"train {len(train_ds)} clips | val {len(val_ds)} clips | world {world}")

    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True)
        if world > 1 else None
    )
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=train_sampler is None,
        sampler=train_sampler, num_workers=args.workers, collate_fn=collate,
        pin_memory=True, drop_last=True, persistent_workers=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=max(2, args.workers // 2), collate_fn=collate, pin_memory=True,
    )

    refs = {r["clip_id"]: [(e["onset"], e["offset"]) for e in r["events"]] for r in val_ds.records}
    durations = {r["clip_id"]: r["duration"] for r in val_ds.records}

    channels = tuple(int(c) for c in args.channels.split(","))
    if bool(args.sed_encoder) != bool(args.sed_ckpt):
        raise ValueError("--sed-encoder and --sed-ckpt must be supplied together")
    if args.sed_ckpt:
        if args.beats_ckpt or args.atst_ckpt or args.speech_ckpt:
            raise ValueError("--sed-ckpt cannot be combined with the BEATs, ATST, or speech branches")
        aug = dict(fms_prob=args.fms_prob, fms_alpha=args.fms_alpha,
                   eq_prob=args.eq_prob, eq_max_db=args.eq_max_db, epn_hidden=args.epn_hidden, conv_heads=args.conv_heads, bdist_k=args.bdist_k, epn_bins=args.epn_bins, frontend=args.frontend, occ_hidden=args.occ_hidden,
                   in_ch=2 if args.residual_root else 1)
        make = lambda: CRNNSedEnc(
            channels=channels, rnn_hidden=args.rnn_hidden, dropout=args.dropout,
            sed_encoder=args.sed_encoder, sed_ckpt=args.sed_ckpt, sed_dim=args.sed_dim,
            freeze_sed=not args.unfreeze_sed, **aug,
        )
    elif args.atst_ckpt:
        if args.beats_ckpt or args.speech_ckpt:
            raise ValueError("--atst-ckpt cannot be combined with the BEATs or speech branches")
        aug = dict(fms_prob=args.fms_prob, fms_alpha=args.fms_alpha,
                   eq_prob=args.eq_prob, eq_max_db=args.eq_max_db, epn_hidden=args.epn_hidden, conv_heads=args.conv_heads, bdist_k=args.bdist_k, epn_bins=args.epn_bins, frontend=args.frontend, occ_hidden=args.occ_hidden,
                   in_ch=2 if args.residual_root else 1)
        make = lambda: CRNNAtst(
            channels=channels, rnn_hidden=args.rnn_hidden, dropout=args.dropout,
            atst_ckpt=args.atst_ckpt, atst_dim=args.atst_dim,
            freeze_atst=not args.unfreeze_atst, **aug,
        )
    elif args.beats_ckpt and args.speech_ckpt:
        aug = dict(fms_prob=args.fms_prob, fms_alpha=args.fms_alpha,
                   eq_prob=args.eq_prob, eq_max_db=args.eq_max_db, epn_hidden=args.epn_hidden, conv_heads=args.conv_heads, bdist_k=args.bdist_k, epn_bins=args.epn_bins, frontend=args.frontend, occ_hidden=args.occ_hidden,
                   in_ch=2 if args.residual_root else 1)
        make = lambda: CRNNBeatsSpeech(
            channels=channels, rnn_hidden=args.rnn_hidden, dropout=args.dropout,
            beats_ckpt=args.beats_ckpt, beats_dim=args.beats_dim, speech_ckpt=args.speech_ckpt,
            speech_dim=args.speech_dim, freeze_beats=not args.unfreeze_beats, **aug,
        )
    elif args.beats_ckpt:
        aug = dict(fms_prob=args.fms_prob, fms_alpha=args.fms_alpha,
                   eq_prob=args.eq_prob, eq_max_db=args.eq_max_db, epn_hidden=args.epn_hidden, conv_heads=args.conv_heads, bdist_k=args.bdist_k, epn_bins=args.epn_bins, frontend=args.frontend, occ_hidden=args.occ_hidden,
                   in_ch=2 if args.residual_root else 1)
        make = lambda: CRNNBeats(
            channels=channels, rnn_hidden=args.rnn_hidden, dropout=args.dropout,
            beats_ckpt=args.beats_ckpt, beats_dim=args.beats_dim,
            freeze_beats=not args.unfreeze_beats, **aug,
        )
    else:
        make = lambda: CRNN(channels=channels, rnn_hidden=args.rnn_hidden,
                            dropout=args.dropout, fms_prob=args.fms_prob,
                            fms_alpha=args.fms_alpha, eq_prob=args.eq_prob,
                            eq_max_db=args.eq_max_db, epn_hidden=args.epn_hidden, conv_heads=args.conv_heads, bdist_k=args.bdist_k, epn_bins=args.epn_bins, frontend=args.frontend, occ_hidden=args.occ_hidden,
                            in_ch=2 if args.residual_root else 1)
    model = make().to(device)
    ema = make().to(device)

    if args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        # warm-start from the EMA weights: they are the ones the val score was measured on
        state = ckpt["ema"] if "ema" in ckpt else ckpt["model"]
        if args.residual_root:
            # single-channel checkpoint into a two-channel model: the residual channel starts at
            # zero weight so the warm start is exactly the old model
            state = dict(state)
            for k, v in list(state.items()):
                if k == "conv.0.conv.0.weight" or (k.startswith("conv.0.") and v.ndim == 4 and v.shape[1] == 1):
                    state[k] = torch.cat([v, torch.zeros_like(v)], dim=1)
                if k.startswith("input_norm.") and v.ndim == 1 and v.shape[0] == 1:
                    state[k] = v.repeat(2)
        want = int(model.input_norm.num_features) if hasattr(model, "input_norm") else None
        for k, v in list(state.items()):
            # extra input channels (PCEN) start at zero weight so the warm start is exactly the parent
            if k.startswith("conv.0.") and v.ndim == 4 and want and v.shape[1] < want:
                state[k] = torch.cat([v, torch.zeros(v.shape[0], want - v.shape[1], *v.shape[2:], dtype=v.dtype)], dim=1)
            if k.startswith("input_norm.") and v.ndim == 1 and want and v.shape[0] < want:
                fill = torch.ones if k.endswith(("weight", "running_var")) else torch.zeros
                state[k] = torch.cat([v, fill(want - v.shape[0], dtype=v.dtype)])
        if (args.speech_ckpt or args.atst_ckpt or args.sed_ckpt) and not any(
            k.startswith(("speech", "atst_proj", "sed_proj")) for k in state
        ):
            # A fused encoder branch widens the GRU input, so recurrence weights restart.
            state = {k: v for k, v in state.items() if not k.startswith("rnn.")}
        if args.sed_ckpt and ckpt.get("args", {}).get("sed_encoder") == args.sed_encoder:
            # same SED encoder family: keep its fine-tuned weights and projection (dropping them restarted
            # every fine-tune of an unfrozen member from the pretrained encoder)
            state = {k: v for k, v in state.items() if not k.startswith(("beats", "atst"))}
        elif args.sed_ckpt:
            state = {k: v for k, v in state.items() if not k.startswith(("beats", "atst", "sed"))}
        elif args.atst_ckpt:
            # ATST replaces BEATs: the old branch and its projection have no counterpart
            state = {k: v for k, v in state.items() if not k.startswith("beats")}
        elif args.beats_ckpt and ckpt.get("args", {}).get("beats_ckpt") not in (None, args.beats_ckpt):
            # a different pretrained BEATs: keep the new encoder's own weights, not the checkpoint's frozen copy
            state = {k: v for k, v in state.items() if not k.startswith("beats.")}
            log("  BEATs weights taken from --beats-ckpt, not the warm-start checkpoint")
        if args.conv_heads and "head_alt.weight" not in state and "head.weight" in state:
            # convention heads start as copies of the main heads
            state = dict(state)
            for src, dst in (("head.", "head_alt."), ("head_bnd.", "head_bnd_alt.")):
                for k, v in list(state.items()):
                    if k.startswith(src):
                        state[dst + k[len(src):]] = v.clone()
        missing, unexpected = model.load_state_dict(state, strict=False)
        fresh = ("occ_", "head_epn.", "head_l2.", "speech", "atst", "sed", "rnn.", "head_alt.", "head_bnd_alt.", "head_bdist.", "pcen.")
        if not any(k.startswith("beats.") for k in state):
            fresh += ("beats.",)
        bad = [k for k in missing if not k.startswith(fresh)]
        if bad or unexpected:
            raise RuntimeError(f"init_from mismatch: missing {bad} unexpected {unexpected}")
        if missing:
            log(f"  new heads initialised fresh: {sorted({k.split('.')[0] for k in missing})}")
        log(f"warm-started from {args.init_from} "
            f"(val Combined {ckpt.get('combined', float('nan')):.4f})")

    ema.load_state_dict(model.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)

    ddp = (DistributedDataParallel(model, device_ids=[local],
                                   find_unused_parameters=bool(args.beats_ckpt or args.atst_ckpt or args.sed_ckpt))
           if world > 1 else model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    log(f"trainable {sum(p.numel() for p in trainable)/1e6:.1f}M of "
        f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    # a fine-tuned transformer gets a lower rate than the heads (DCASE 2024 stage-2 recipe)
    beats_params = [(n, p) for n, p in model.named_parameters()
                    if p.requires_grad and n.startswith(("beats.", "atst.", "sed."))]
    other_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and not n.startswith(("beats.", "atst.", "sed."))]
    groups = [{"params": other_params, "lr": args.lr}]
    if beats_params:
        # layer-wise learning-rate decay (DCASE 2024 stage-2 recipe): the top transformer block gets
        # lr * beats_lr_scale, each block below it a further factor of beats_layer_decay
        import re
        layer_re = r"(?:encoder\.layers|blocks)\.(\d+)\."
        n_layers = 1 + max((int(m.group(1)) for n, _ in beats_params
                            for m in [re.search(layer_re, n)] if m), default=-1)
        by_depth = {}
        for n, p in beats_params:
            m = re.search(layer_re, n)
            depth = int(m.group(1)) if m else (n_layers if "encoder.layer_norm" in n else -1)
            by_depth.setdefault(depth, []).append(p)
        for depth, ps in by_depth.items():
            scale = args.beats_lr_scale * (args.beats_layer_decay ** (n_layers - depth))
            groups.append({"params": ps, "lr": args.lr * scale})
        log(f"BEATs unfrozen: {sum(p.numel() for _, p in beats_params)/1e6:.1f}M params, {n_layers} blocks, "
            f"top lr x{args.beats_lr_scale}, layer decay {args.beats_layer_decay}")
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=1e-2)

    steps_per_epoch = len(train_dl)
    total_steps = args.epochs * steps_per_epoch

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return step / max(args.warmup, 1)
        prog = (step - args.warmup) / max(total_steps - args.warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    track_decode = dict(threshold=0.5, median_k=5, merge_gap=0.0, min_dur=0.05)
    track_epn = dict(p_min=0.3, conf_min=0.6, k=20) if args.epn_weight > 0 or args.collar_weight > 0 else None

    best, step, t0 = -1.0, 0, time.time()
    running = []

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch in train_dl:
            for k in ("wav", "target", "target_cat", "target_onset", "target_offset",
                      "target_l2", "target_epn", "mask_epn", "event_weight", "frame_mask",
                      "clip_cat", "clip_act", "has_frames", "has_cats", "bnd_rel", "tier", "collar_mask"):
                batch[k] = batch[k].to(device, non_blocking=True)
            if batch.get("res") is not None:
                batch["res"] = batch["res"].to(device, non_blocking=True)
            if args.mixup_prob > 0:
                batch = mixup(batch, args.mixup_alpha, args.mixup_prob)

            # per-tier loss weights: gold 1.0, silver as configured, bronze_pseudo
            # lowest because its timestamps are inferred rather than annotated
            tier = batch["tier"]
            clip_w = torch.ones_like(tier, dtype=torch.float32)
            clip_w = torch.where(tier == 1, torch.full_like(clip_w, args.silver_weight), clip_w)
            clip_w = torch.where(tier == 3, torch.full_like(clip_w, args.bronze_pseudo_weight), clip_w)
            clip_w = torch.where(tier == 4, torch.full_like(clip_w, args.parent_pseudo_weight), clip_w)
            clip_w = torch.where(tier == 6, torch.full_like(clip_w, args.synth_weight), clip_w)
            bg_w = torch.ones_like(clip_w)
            bg_w = torch.where(tier == 1, torch.full_like(bg_w, args.silver_bg_weight), bg_w)
            bg_w = torch.where(tier == 3, torch.full_like(bg_w, args.silver_bg_weight), bg_w)
            bg_w = torch.where(tier == 4, torch.full_like(bg_w, args.silver_bg_weight), bg_w)
            n_fr = int(batch["frame_mask"].shape[1])

            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits, logits_cat, log_on, log_off, log_l2, epn = ddp(batch["wav"], n_fr, batch.get("res"))
                if args.consistency_weight > 0:
                    ema.eval()
                    with torch.no_grad():
                        t_logits, _, t_on, t_off, _, t_epn = ema(batch["wav"], n_fr, batch.get("res"))
                    ema.train()
                    s_logits, s_on, s_off, s_epn, c_shift = logits, log_on, log_off, epn, 0
                    if args.consistency_aug:
                        wav_v, res_v, c_shift = consistency_view(batch["wav"], batch.get("res"), args)
                        s_logits, _, s_on, s_off, _, s_epn = ddp(wav_v, n_fr, res_v)
            logits, logits_cat = logits.float(), logits_cat.float()
            log_on, log_off, log_l2 = log_on.float(), log_off.float(), log_l2.float()
            logits_s, log_on_s, log_off_s = logits, log_on, log_off
            if args.conv_heads:
                # Gold samples supervise the main heads, every other tier its own convention heads
                alt = (ddp.module if hasattr(ddp, "module") else model)._alt
                gold = (tier == 0).view(-1, 1)
                logits_s = torch.where(gold, logits, alt[0].float())
                log_on_s = torch.where(gold, log_on, alt[1].float())
                log_off_s = torch.where(gold, log_off, alt[2].float())

            fmask = batch["frame_mask"] * batch["has_frames"].view(-1, 1)
            ev_w = batch["event_weight"] if args.event_norm else None
            fmask_act = fmask
            if args.pu_near_sec > 0:
                # Silver / pseudo negatives next to a labelled event are where fragmented long events leave false
                # negatives: treat them as unlabelled (low weight) instead of trusted background; Gold stays exact
                k = int(round(args.pu_near_sec / 0.01))
                pos = (batch["target"] > 0.5).float()
                near = F.max_pool1d(pos[:, None], 2 * k + 1, stride=1, padding=k)[:, 0] * (1 - pos)
                soft = near * (tier != 0).float()[:, None]
                fmask_act = fmask * (1 - soft * (1 - args.pu_near_weight))
            l_frame = frame_loss(logits_s, batch["target"], fmask_act, clip_w, bg_w, ev_w)

            bnd_w = clip_w * batch["bnd_rel"]
            bnd_w = torch.where(tier == 4, bnd_w * args.parent_bnd_weight, bnd_w)
            ones = torch.ones_like(clip_w)
            l_bnd = (
                frame_loss(log_on_s, batch["target_onset"], fmask, bnd_w, ones)
                + frame_loss(log_off_s, batch["target_offset"], fmask, bnd_w, ones)
            )
            cat_w = clip_w * batch["has_cats"]
            l_aux = frame_loss(
                logits_cat.transpose(1, 2).reshape(-1, n_fr),
                batch["target_cat"].transpose(1, 2).reshape(-1, n_fr),
                fmask.repeat_interleave(logits_cat.shape[-1], dim=0),
                cat_w.repeat_interleave(logits_cat.shape[-1]),
                bg_w.repeat_interleave(logits_cat.shape[-1]),
            )

            # layer-2 channel: the frames where a second, overlapping event is active.
            # 30.6% of event pairs in this corpus overlap and channel 1 structurally
            # cannot express that.  Supervised on the same tiers as channel 1 - Bronze
            # has no timestamps, so it cannot say whether two events overlap either.
            l2_mask = fmask * (tier == 0).float()[:, None] if args.l2_gold_only else fmask
            l_l2 = frame_loss(log_l2, batch["target_l2"], l2_mask, clip_w, ones)

            # clip-level MIL - the only path through which Bronze contributes
            probs_cat = torch.sigmoid(logits_cat)
            clip_pred = linear_softmax_pool(probs_cat, batch["frame_mask"])
            l_clip = F.binary_cross_entropy(
                clip_pred.clamp(1e-6, 1 - 1e-6), batch["clip_cat"], reduction="none"
            ).mean(1)
            l_clip = (l_clip * batch["has_cats"]).sum() / batch["has_cats"].sum().clamp(min=1.0)

            loss = (
                l_frame
                + args.aux_weight * l_aux
                + args.clip_weight * l_clip
                + args.bnd_weight * l_bnd
                + args.l2_weight * l_l2
            )
            if args.long_weight > 0:
                # Gold frames inside positive runs of at least long_min_sec: the long events the fusion half-misses
                lm = long_run_mask(batch["target"], int(round(args.long_min_sec / 0.01))) * fmask * (torch.ones_like(tier) if args.long_all_tiers else (tier == 0)).float()[:, None]
                l_long = F.binary_cross_entropy_with_logits(logits, batch["target"], reduction="none")
                loss = loss + args.long_weight * (l_long * lm).sum() / lm.sum().clamp(min=1.0)
            if args.long_bnd_weight > 0:
                # the edges of long Gold events, where long-event extent is lost
                lr = long_run_mask(batch["target"], int(round(args.long_min_sec / 0.01)))
                edge = (F.max_pool1d(lr[:, None], 41, stride=1, padding=20)[:, 0] - F.avg_pool1d(lr[:, None], 3, stride=1, padding=1)[:, 0].floor()).clamp(min=0)
                bm = edge * fmask * (tier == 0).float()[:, None]
                l_lb = F.binary_cross_entropy_with_logits(log_on_s, batch["target_onset"], reduction="none") + F.binary_cross_entropy_with_logits(log_off_s, batch["target_offset"], reduction="none")
                loss = loss + args.long_bnd_weight * (l_lb * bm).sum() / bm.sum().clamp(min=1.0)
            if args.smooth_weight > 0:
                # over-segmentation penalty: frame-to-frame flicker of the activity log-probabilities, except at label edges
                lp = F.logsigmoid(logits_s); lq = F.logsigmoid(-logits_s)
                d = (lp[:, 1:] - lp[:, :-1]).abs().clamp(max=args.smooth_tau) ** 2 + (lq[:, 1:] - lq[:, :-1]).abs().clamp(max=args.smooth_tau) ** 2
                tgt = (batch["target"] > 0.5).float()
                edge = F.max_pool1d((tgt[:, 1:] - tgt[:, :-1]).abs()[:, None], 11, stride=1, padding=5)[:, 0]
                sm_mask = fmask[:, 1:] * fmask[:, :-1] * (1 - edge)
                loss = loss + args.smooth_weight * (d * sm_mask).sum() / sm_mask.sum().clamp(min=1.0)
            if args.short_weight > 0:
                # Gold frames inside runs up to short_max_sec: the sub-second events that hold 59% of the Dice deficit
                sm = short_run_mask(batch["target"], int(round(args.short_max_sec / 0.01))) * fmask * (tier == 0).float()[:, None]
                l_short = F.binary_cross_entropy_with_logits(logits, batch["target"], reduction="none")
                loss = loss + args.short_weight * (l_short * sm).sum() / sm.sum().clamp(min=1.0)
            if args.dice_weight > 0:
                # soft Dice per clip on the activity head, over labelled frames only
                if args.dice_thr > 0:
                    # the metric thresholds the curve and median-filters it; match that instead of scoring raw probabilities
                    thr_logit = math.log(args.dice_thr / (1 - args.dice_thr))
                    pr = torch.sigmoid((logits - thr_logit) / max(args.dice_tau, 1e-3))
                else:
                    pr = torch.sigmoid(logits)
                if args.dice_medk > 1:
                    k = int(args.dice_medk) | 1
                    pr = F.avg_pool1d(pr.unsqueeze(1), k, stride=1, padding=k // 2).squeeze(1)
                pr = pr * fmask
                tg = batch["target"] * fmask
                if args.dice_alpha == 0.5:
                    inter = (pr * tg).sum(1); sm_c = args.dice_smooth
                    l_dice = (1 - (2 * inter + sm_c) / (pr.sum(1) + tg.sum(1) + sm_c))
                else:
                    # Tversky: the fusion over-predicts frames on the short-event clips that hold 59% of the Dice deficit
                    tp = (pr * tg).sum(1); fp = (pr * (1 - tg)).sum(1); fn = ((1 - pr) * tg).sum(1)
                    a = args.dice_alpha
                    l_dice = 1 - (tp + args.dice_smooth / 2) / (tp + a * fp + (1 - a) * fn + args.dice_smooth / 2)
                # tier weights, so pseudo-labelled parent clips do not shape the Dice term at full strength
                dw = batch["has_frames"] * clip_w
                l_dice = (l_dice * dw).sum() / dw.sum().clamp(min=1.0)
                loss = loss + args.dice_weight * l_dice
            if args.bdl_weight > 0 or args.edge_margin_weight > 0:
                pos, d_out, d_in = edge_distances(batch["target"])
            if args.bdl_weight > 0:
                phi = torch.where(pos, -d_in, d_out).clamp(-args.bdl_clip, args.bdl_clip) / args.bdl_clip
                l_bdl = (torch.sigmoid(logits) * phi * fmask).sum(1) / fmask.sum(1).clamp(min=1.0)
                dw = batch["has_frames"] * clip_w
                loss = loss + args.bdl_weight * (l_bdl * dw).sum() / dw.sum().clamp(min=1.0)
            if args.edge_margin_weight > 0:
                thr_logit = math.log(args.edge_thr / (1 - args.edge_thr))
                band_in = pos & (d_in <= args.edge_band)
                band_out = ~pos & (d_out <= args.edge_band)
                em = fmask * (tier == 0).float()[:, None]
                hinge = torch.where(band_in, F.relu(args.edge_margin - (logits - thr_logit)),
                                    F.relu(args.edge_margin - (thr_logit - logits)))
                bm = (band_in | band_out).float() * em
                loss = loss + args.edge_margin_weight * (hinge * bm).sum() / bm.sum().clamp(min=1.0)
            if args.ohem_weight > 0:
                om = fmask * (tier == 0).float()[:, None]
                l_px = F.binary_cross_entropy_with_logits(logits, batch["target"], reduction="none") * om
                srt = l_px.sort(1, descending=True).values
                k = (om.sum(1) * args.ohem_frac).ceil().clamp(min=1.0)
                keep = (torch.arange(srt.shape[1], device=device)[None] < k[:, None]).float()
                has = (om.sum(1) > 0).float()
                l_ohem = ((srt * keep).sum(1) / k * has).sum() / has.sum().clamp(min=1.0)
                loss = loss + args.ohem_weight * l_ohem
            l_cons = torch.zeros((), device=device)
            l_cons_epn = torch.zeros((), device=device)
            if args.consistency_weight > 0:
                # teacher targets from the EMA model on the same (mixed) input; the student's
                # dropout/augmentation supplies the perturbation, and Bronze frames count too
                k, T = c_shift, n_fr
                fm = batch["frame_mask"][:, k:]
                diff = (torch.sigmoid(s_logits.float()[:, :T - k]) - torch.sigmoid(t_logits.float()[:, k:])) ** 2
                diff = diff + (torch.sigmoid(s_on.float()[:, :T - k]) - torch.sigmoid(t_on.float()[:, k:])) ** 2
                diff = diff + (torch.sigmoid(s_off.float()[:, :T - k]) - torch.sigmoid(t_off.float()[:, k:])) ** 2
                l_cons = (diff * fm).sum() / fm.sum().clamp(min=1.0)
                ramp = min(1.0, step / max(args.consistency_rampup, 1))
                loss = loss + args.consistency_weight * ramp * l_cons
                if args.consistency_epn_weight > 0:
                    # first-slot interval regressions, weighted by the teacher's activity on the aligned frames
                    m_epn = torch.stack([torch.sigmoid(t_logits.float()[:, k:]) * fm, torch.zeros_like(fm)], dim=-1)
                    l_cons_epn = epn_loss(s_epn.float()[:, :T - k], t_epn.float()[:, k:].detach(), m_epn)
                    loss = loss + args.consistency_epn_weight * ramp * l_cons_epn
            l_epn = torch.zeros((), device=device)
            if args.epn_weight > 0:
                l_epn = epn_loss(epn.float(), batch["target_epn"],
                                 batch["mask_epn"] * fmask[:, :, None], bnd_w)
                loss = loss + args.epn_weight * l_epn
            l_dfl = torch.zeros((), device=device)
            if args.epn_bins > 0:
                # distribution focal loss on the Gold frames' true distances, below the bin ceiling
                lg = (ddp.module if hasattr(ddp, "module") else model).head_epn._dist_logits.float()
                tgt = batch["target_epn"]
                m = batch["mask_epn"] * fmask[:, :, None]
                m = m.repeat_interleave(2, dim=-1) * (tier == 0).view(-1, 1, 1).float()
                m = m * (tgt < (args.epn_bins - 1) * 0.01).float()
                if m.sum() > 0:
                    sb = (tgt / 0.01).clamp(0, args.epn_bins - 1 - 1e-4)
                    lo = sb.floor(); w = sb - lo
                    logp = lg.log_softmax(-1)
                    lo_l = lo.long()
                    nll = -((1 - w) * logp.gather(-1, lo_l[..., None]).squeeze(-1) + w * logp.gather(-1, (lo_l + 1)[..., None]).squeeze(-1))
                    l_dfl = (nll * m).sum() / m.sum()
                    loss = loss + args.epn_dfl_weight * l_dfl
            l_bdist = torch.zeros((), device=device)
            if args.bdist_k > 0:
                # distribution focal loss: each Gold frame within K frames of an edge learns the
                # fractional bin of that edge (linear interpolation between the two nearest bins)
                K = args.bdist_k
                lg = (ddp.module if hasattr(ddp, "module") else model)._bdist.float()
                bi, ti, si, lo_b, w_lo = [], [], [], [], []
                for b, evs in enumerate(batch["collar_events"]):
                    if int(tier[b]) != 0 or not evs:
                        continue
                    T = int(batch["frame_mask"][b].sum().item())
                    centres = (torch.arange(T, device=device, dtype=torch.float32) + 0.5) * FRAME_SEC
                    for side in (0, 1):
                        edges = torch.tensor([e[side] for e in evs], device=device, dtype=torch.float32)
                        rel = (edges[None, :] - centres[:, None]) / FRAME_SEC
                        near = rel.abs().argmin(1)
                        o = rel[torch.arange(T, device=device), near]
                        keep = o.abs() <= K
                        if not bool(keep.any()):
                            continue
                        o = o[keep] + K
                        lo = o.floor().clamp(0, 2 * K - 1)
                        bi.append(torch.full_like(lo, b, dtype=torch.long)); ti.append(keep.nonzero().squeeze(1))
                        si.append(torch.full_like(lo, side, dtype=torch.long)); lo_b.append(lo.long()); w_lo.append(1.0 - (o - lo))
                if bi:
                    bi, ti, si, lo_b, w_lo = (torch.cat(x) for x in (bi, ti, si, lo_b, w_lo))
                    logp = lg[bi, ti, si].log_softmax(-1)
                    l_bdist = -(w_lo * logp.gather(1, lo_b[:, None]).squeeze(1) + (1.0 - w_lo) * logp.gather(1, (lo_b + 1)[:, None]).squeeze(1)).mean()
            l_collar = torch.zeros((), device=device)
            if args.collar_weight > 0:
                if args.collar_v2:
                    sig = 1.0 - 0.65 * min(1.0, step / max(args.collar_anneal, 1))
                    l_collar = collar_loss_v2(
                        epn.float(), logits, batch["collar_events"], batch["frame_mask"],
                        (tier == 0) & batch["collar_mask"].bool(), sigma_scale=sig, decode_cfg=track_epn)
                else:
                    l_collar = collar_loss(
                        epn.float(), logits, batch["collar_events"], batch["frame_mask"],
                        (tier == 0) & batch["collar_mask"].bool(),
                    )
                ramp = min(1.0, step / max(args.collar_rampup, 1))
                loss = loss + args.collar_weight * ramp * l_collar
            if args.bdist_k > 0:
                loss = loss + args.bdist_weight * l_bdist
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

            with torch.no_grad():
                d = args.ema_decay
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(d).add_(pm.detach(), alpha=1 - d)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)

            running.append(loss.item())
            step += 1

            if step % args.log_every == 0:
                rate = step / (time.time() - t0)
                log(
                    f"ep {epoch} step {step}/{total_steps} loss {np.mean(running[-args.log_every:]):.4f} "
                    f"frame {l_frame.item():.4f} aux {l_aux.item():.4f} clip {l_clip.item():.4f} "
                    f"bnd {l_bnd.item():.4f} epn {l_epn.item():.4f} collar {l_collar.item():.4f} bdist {l_bdist.item():.4f} dfl {l_dfl.item():.4f} cons {l_cons.item():.4f} consp {l_cons_epn.item():.4f} "
                    f"lr {sched.get_last_lr()[0]:.2e} {rate:.2f} it/s "
                    f"eta {(total_steps - step) / max(rate, 1e-6) / 60:.1f}m"
                )

            if step % args.eval_every == 0 and is_main():
                s = evaluate(ema, val_dl, device, track_decode, track_epn, refs, durations, step)
                if s.combined > best:
                    best = s.combined
                    torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                                "args": vars(args), "step": step, "combined": best},
                               out_dir / "best.pt")
                    log(f"  saved best.pt (Combined {best:.4f})")

    if is_main():
        s = evaluate(ema, val_dl, device, track_decode, track_epn, refs, durations, "final")
        log(f"\nFINAL EMA full-val: {json.dumps(s.as_dict())}")
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                    "args": vars(args), "combined": s.combined}, out_dir / "last.pt")
        if s.combined > best:
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                        "args": vars(args), "combined": s.combined}, out_dir / "best.pt")
            log(f"saved best.pt (Combined {s.combined:.4f})")
        log(f"done in {(time.time() - t0) / 60:.1f} min")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
