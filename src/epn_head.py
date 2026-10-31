"""Event Proposal Network head: per-frame time-since-onset / time-to-offset regression.

Follows Schmid et al. 2026 (arXiv:2601.04178, "Boundary-Aware Optimization and Inference"):
every active frame proposes the complete interval of the event covering it, trained with a
duration-normalised IoU loss, and events are read off by greedy proposal selection instead
of thresholding an activity curve.  Two proposal layers follow the existing two-colouring
so a frame inside two overlapping events regresses both.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from overlap_head import two_colour

FRAME_SEC = 0.01
N_LAYERS = 2


def epn_targets(events: list[dict], n_frames: int, offset_sec: float = 0.0
                ) -> tuple[np.ndarray, np.ndarray]:
    """(T, 4) distances [on1, off1, on2, off2] in seconds and a (T, 2) frame mask."""
    tgt = np.zeros((n_frames, 2 * N_LAYERS), dtype=np.float32)
    mask = np.zeros((n_frames, N_LAYERS), dtype=np.float32)
    if n_frames == 0:
        return tgt, mask
    tc = (np.arange(n_frames, dtype=np.float64) + 0.5) * FRAME_SEC + offset_sec
    for layer, spans in enumerate(two_colour(events)):
        for on, off in spans:
            idx = np.flatnonzero((tc >= on) & (tc < off))
            if idx.size == 0:
                continue
            tgt[idx, 2 * layer] = tc[idx] - on
            tgt[idx, 2 * layer + 1] = off - tc[idx]
            mask[idx, layer] = 1.0
    return tgt, mask


def epn_loss(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor,
             clip_weight: torch.Tensor | None = None) -> torch.Tensor:
    """Duration-normalised 1-IoU over active frames, per layer (paper eq. 4)."""
    total = pred.new_zeros(())
    for layer in range(N_LAYERS):
        d_on, d_off = tgt[..., 2 * layer], tgt[..., 2 * layer + 1]
        p_on, p_off = pred[..., 2 * layer], pred[..., 2 * layer + 1]
        inter = torch.minimum(d_on, p_on) + torch.minimum(d_off, p_off)
        union = d_on + d_off + p_on + p_off - inter
        iou = inter / union.clamp(min=1e-6)
        dur = (d_on + d_off).clamp(min=FRAME_SEC)
        m = mask[..., layer]
        if clip_weight is not None:
            m = m * clip_weight.view(-1, 1)
        total = total + ((1.0 - iou) / dur * m).sum() / m.sum().clamp(min=1.0)
    return total


def collar_loss(
    distances: torch.Tensor,
    presence_logits: torch.Tensor,
    references: list[list[tuple[float, float]]],
    frame_mask: torch.Tensor,
    gold_mask: torch.Tensor,
    top_k: int = 20,
) -> torch.Tensor:
    """Hard-Hungarian soft collar-F1 loss over the highest-presence EPN proposals."""
    losses = []
    for b, refs in enumerate(references):
        if not bool(gold_mask[b]) or not refs:
            continue
        n_frames = int(frame_mask[b].sum().item())
        if n_frames == 0:
            continue
        presence = torch.sigmoid(presence_logits[b, :n_frames])
        k = min(top_k, n_frames)
        proposal_presence, frames = torch.topk(presence, k)
        centres = (frames.to(distances.dtype) + 0.5) * FRAME_SEC
        d_on = distances[b, frames, 0]
        d_off = distances[b, frames, 1]
        starts = (centres - d_on).clamp(0.0, n_frames * FRAME_SEC)
        ends = (centres + d_off).clamp(0.0, n_frames * FRAME_SEC)

        ref = distances.new_tensor(refs)
        ref_dur = (ref[:, 1] - ref[:, 0]).clamp(min=0.0)
        collar = torch.maximum(0.2 * ref_dur, ref_dur.new_full(ref_dur.shape, 0.05))
        tau = torch.maximum(0.25 * collar, collar.new_full(collar.shape, 0.01))
        log_q = (
            F.logsigmoid((collar[:, None] - (starts[None] - ref[:, 0, None]).abs()) / tau[:, None])
            + F.logsigmoid((collar[:, None] - (ends[None] - ref[:, 1, None]).abs()) / tau[:, None])
        )
        q = log_q.exp()
        # Assignment is deliberately detached; gradients flow only through selected matches.
        from scipy.optimize import linear_sum_assignment
        costs = (-log_q.detach() - proposal_presence.detach().clamp_min(1e-8).log()).cpu().numpy()
        ref_idx, prop_idx = linear_sum_assignment(costs)
        # soft TP must carry the proposal's presence: with a bare q the loss falls without
        # bound as presence is suppressed (the collapse at step 1500 in v51/v58/v59)
        matched_q = (q[ref_idx, prop_idx] * proposal_presence[prop_idx]).sum()
        soft_f1 = 2 * matched_q / (len(refs) + proposal_presence.sum()).clamp_min(1e-8)
        losses.append(1 - soft_f1)
    return torch.stack(losses).mean() if losses else distances.new_zeros(())


def collar_loss_v2(
    distances: torch.Tensor,
    presence_logits: torch.Tensor,
    references: list[list[tuple[float, float]]],
    frame_mask: torch.Tensor,
    gold_mask: torch.Tensor,
    sigma_scale: float = 1.0,
    decode_cfg: dict | None = None,
) -> torch.Tensor:
    """Soft collar-F1 over the proposals the deployed greedy decoder actually emits: the matched
    term is the closed-form collar reward of a Gaussian edge policy (sigma = sigma_scale x collar,
    annealed by the caller) weighted by the proposal's mean presence, and the denominator counts
    emitted proposals rather than top-k frames."""
    cfg = dict(p_min=0.3, conf_min=0.6, k=20)
    if decode_cfg:
        cfg.update(decode_cfg)
    losses = []
    for b, refs in enumerate(references):
        if not bool(gold_mask[b]) or not refs:
            continue
        n_frames = int(frame_mask[b].sum().item())
        if n_frames == 0:
            continue
        presence = torch.sigmoid(presence_logits[b, :n_frames])
        d_on_all, d_off_all = distances[b, :n_frames, 0], distances[b, :n_frames, 1]
        with torch.no_grad():
            spans = epn_decode(presence.detach().float().cpu().numpy(), d_on_all.detach().float().cpu().numpy(),
                               d_off_all.detach().float().cpu().numpy(), duration=n_frames * FRAME_SEC, **cfg)
        if not spans:
            losses.append(presence.new_ones(()))
            continue
        # the frame that owns each emitted proposal is its highest-presence frame inside the span
        frames, p_j = [], []
        for a, e in spans:
            lo, hi = max(0, int(a / FRAME_SEC)), min(n_frames, max(int(a / FRAME_SEC) + 1, int(e / FRAME_SEC)))
            seg = presence[lo:hi]
            frames.append(lo + int(seg.detach().argmax().item()))
            p_j.append(seg.mean())
        frames_t = torch.tensor(frames, device=distances.device)
        p_j = torch.stack(p_j)
        centres = (frames_t.to(distances.dtype) + 0.5) * FRAME_SEC
        starts = (centres - d_on_all[frames_t]).clamp(0.0, n_frames * FRAME_SEC)
        ends = (centres + d_off_all[frames_t]).clamp(0.0, n_frames * FRAME_SEC)
        ref = distances.new_tensor(refs)
        ref_dur = (ref[:, 1] - ref[:, 0]).clamp(min=0.0)
        collar = torch.maximum(0.2 * ref_dur, ref_dur.new_full(ref_dur.shape, 0.05))
        sigma = (sigma_scale * collar).clamp_min(1e-3)

        def mass(d):  # P(|edge error| <= collar) under N(d, sigma^2)
            return torch.special.ndtr((collar[:, None] - d) / sigma[:, None]) - torch.special.ndtr((-collar[:, None] - d) / sigma[:, None])

        e_ij = mass(starts[None] - ref[:, 0, None]) * mass(ends[None] - ref[:, 1, None])
        from scipy.optimize import linear_sum_assignment
        costs = (-(e_ij.detach() + 1e-8).log() - (p_j.detach() + 1e-8).log()).cpu().numpy()
        ref_idx, prop_idx = linear_sum_assignment(costs)
        soft_tp = (e_ij[ref_idx, prop_idx] * p_j[prop_idx]).sum()
        soft_f1 = 2 * soft_tp / (len(refs) + p_j.sum()).clamp_min(1e-8)
        losses.append(1 - soft_f1)
    return torch.stack(losses).mean() if losses else distances.new_zeros(())


BIN_SEC = 0.01


class EPNHead(nn.Module):
    def __init__(self, d_in: int, hidden: int = 128, layers: int = 2, dropout: float = 0.1, n_bins: int = 0):
        super().__init__()
        self.gru = nn.GRU(d_in, hidden, num_layers=layers, batch_first=True,
                          bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.out = nn.Linear(2 * hidden, 2 * N_LAYERS)
        # distributional distances (GFL / TriDet style): a categorical over n_bins x BIN_SEC per
        # distance channel, decoded at inference by the collar-matched rule (argmax of the mass
        # inside the metric's tolerance box) instead of the regressed point estimate
        self.n_bins = n_bins
        # inference switch (EPN_COLLAR_DECODE=1): the tuner compares the collar rule against the
        # regressed distances on identical weights, and checkpoint selection stays on the regression
        self.collar_decode = os.environ.get("EPN_COLLAR_DECODE", "0") == "1"
        if n_bins > 0:
            self.out_dist = nn.Linear(2 * hidden, 2 * N_LAYERS * n_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        d = F.softplus(self.out(h))
        if self.n_bins > 0:
            lg = self.out_dist(h).view(h.shape[0], h.shape[1], 2 * N_LAYERS, self.n_bins)
            self._dist_logits = lg
            if not self.training and self.collar_decode:
                d = self.decode_collar(lg)
        return d

    def decode_collar(self, lg: torch.Tensor) -> torch.Tensor:
        p = lg.float().softmax(-1)
        centres = (torch.arange(self.n_bins, device=lg.device, dtype=torch.float32) + 0.5) * BIN_SEC
        exp = (p * centres).sum(-1)                                    # (B, T, 2L)
        dur = exp[..., 0::2] + exp[..., 1::2]                          # (B, T, L)
        half = (torch.maximum(0.2 * dur, torch.full_like(dur, 0.05)) / BIN_SEC).round().long()
        half = half.repeat_interleave(2, dim=-1).clamp(0, self.n_bins // 2)   # (B, T, 2L)
        cs = torch.cat([torch.zeros_like(p[..., :1]), p.cumsum(-1)], -1)    # (B, T, 2L, n_bins + 1)
        idx = torch.arange(self.n_bins, device=lg.device)
        lo = (idx[None, None, None, :] - half[..., None]).clamp(min=0)
        hi = (idx[None, None, None, :] + half[..., None] + 1).clamp(max=self.n_bins)
        mass = cs.gather(-1, hi) - cs.gather(-1, lo)
        return (mass.argmax(-1).float() + 0.5) * BIN_SEC


def epn_decode(presence: np.ndarray, d_on: np.ndarray, d_off: np.ndarray,
               p_min: float = 0.1, k: int = 15, conf_min: float = 0.5,
               min_dur: float = 0.0, duration: float | None = None,
               vote: bool = False, suppress_accepted_only: bool = False
               ) -> list[tuple[float, float]]:
    """Greedy proposal selection (paper alg. 1): highest-presence frame first, confidence =
    mean presence inside its proposal, proposals overlapping a selected one are dropped.
    `vote` replaces the winning frame's edges with the presence-weighted median of the
    edges proposed by every live frame inside it.  `suppress_accepted_only` stops a proposal
    rejected on confidence from erasing the proposals it overlaps."""
    n = presence.shape[0]
    if n == 0:
        return []
    tc = (np.arange(n, dtype=np.float64) + 0.5) * FRAME_SEC
    starts = tc - d_on
    ends = tc + d_off
    if duration is not None:
        starts = np.clip(starts, 0.0, duration)
        ends = np.clip(ends, 0.0, duration)
    else:
        starts = np.clip(starts, 0.0, None)
    cum = np.concatenate([[0.0], np.cumsum(presence, dtype=np.float64)])
    alive = presence >= p_min
    order = np.argsort(-presence, kind="stable")
    out = []
    for t in order:
        if not alive[t]:
            continue
        if presence[t] < p_min:
            break
        s, e = starts[t], ends[t]
        if vote:
            inside = alive & (tc >= s) & (tc < e)
            if inside.sum() >= 3:
                s = _weighted_median(starts[inside], presence[inside])
                e = _weighted_median(ends[inside], presence[inside])
        a = int(np.floor(s / FRAME_SEC))
        b = max(int(np.ceil(e / FRAME_SEC)), a + 1)
        a, b = max(a, 0), min(b, n)
        conf = float((cum[b] - cum[a]) / max(b - a, 1))
        accept = conf >= conf_min and e - s >= max(min_dur, 1e-3)
        if accept or not suppress_accepted_only:
            alive &= ~((starts < e) & (ends > s))
        else:
            alive[t] = False
        if accept:
            out.append((round(float(s), 4), round(float(e), 4)))
        if len(out) >= k:
            break
    return sorted(out)


def _weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][np.searchsorted(cw, 0.5 * cw[-1])])
