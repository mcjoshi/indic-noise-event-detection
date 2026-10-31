"""Proposal-conditioned boundary refinement, trained on jittered proposals.

The collar sweep is the evidence: natural Event-F1 rises 0.367 -> 0.618 when the
tolerance is tripled, so events are being *found* and their endpoints misplaced.  The
survey also rules out the obvious fix - a 10 ms grid quantises an endpoint by at most
~5 ms, against a 50 ms minimum collar - so higher resolution cannot explain the loss.
What it recommends instead is a head that looks at a *proposed* endpoint and regresses a
correction, trained against continuous targets (BSN 2018, PBRNet 2020, AFSD 2021,
ActionFormer 2022).

This refiner takes a local window of the model's own three curves around a proposed
endpoint - activity, onset probability, offset probability - and predicts a categorical
distribution over displacements in 5 ms bins.  A distribution rather than a single
regressed number because the survey's recommended form is a local displacement
distribution, whose expectation is more stable than an argmax when evidence is flat or
bimodal.

Training proposals are the ground-truth endpoints jittered by up to +/-250 ms, so the
refiner learns to correct an imperfect proposal rather than to re-detect from scratch.
That is the failure mode at inference: the decoder hands it an endpoint that is close but
outside the collar.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

FRAME_SEC = 0.01
HALF_WIN_FRAMES = 25          # +/- 250 ms of context around the proposal
N_BINS = 101                  # -250..+250 ms in 5 ms steps
BIN_MS = 5.0
MAX_DISP_SEC = 0.25


def displacement_bins() -> np.ndarray:
    return np.linspace(-MAX_DISP_SEC, MAX_DISP_SEC, N_BINS)


class BoundaryRefiner(nn.Module):
    """Small 1-D CNN over (activity, onset, offset) curves centred on a proposal."""

    def __init__(self, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        width = 2 * HALF_WIN_FRAMES + 1
        self.net = nn.Sequential(
            nn.Conv1d(3, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(hidden * width, hidden), nn.GELU(),
            nn.Linear(hidden, N_BINS),
        )
        # one refiner serves both endpoint kinds; the kind is supplied as an input channel
        self.kind_embed = nn.Parameter(torch.zeros(2, 3, width))

    def forward(self, window: torch.Tensor, kind: torch.Tensor) -> torch.Tensor:
        x = window + self.kind_embed[kind]
        return self.head(self.net(x))

    @torch.no_grad()
    def expected_shift(self, window: torch.Tensor, kind: torch.Tensor) -> torch.Tensor:
        """Expectation of the predicted displacement distribution, in seconds."""
        probs = torch.softmax(self.forward(window, kind), dim=-1)
        bins = torch.tensor(displacement_bins(), dtype=probs.dtype, device=probs.device)
        return (probs * bins).sum(dim=-1)


def extract_window(
    activity: np.ndarray, onset: np.ndarray, offset: np.ndarray, frame_idx: int
) -> np.ndarray:
    """(3, width) local view, edge-padded so proposals near a clip edge still work."""
    width = 2 * HALF_WIN_FRAMES + 1
    n = activity.shape[0]
    if n == 0:
        return np.zeros((3, width), dtype=np.float32)

    # a proposal can sit beyond the curve entirely (a boundary at the very end of a short
    # clip, jittered outward), which would leave an empty slice that edge-padding cannot
    # extend - clamp the centre into range first
    frame_idx = int(np.clip(frame_idx, 0, n - 1))
    lo, hi = frame_idx - HALF_WIN_FRAMES, frame_idx + HALF_WIN_FRAMES + 1
    pad_lo, pad_hi = max(0, -lo), max(0, hi - n)
    lo, hi = max(lo, 0), min(hi, n)

    out = []
    for curve in (activity, onset, offset):
        seg = curve[lo:hi]
        if pad_lo or pad_hi:
            seg = np.pad(seg, (pad_lo, pad_hi), mode="edge")
        out.append(seg[:width])
    return np.stack(out).astype(np.float32)


def soft_target(displacement_sec: float, sigma_sec: float = 0.015) -> np.ndarray:
    """Gaussian-smoothed target over displacement bins.

    A one-hot target would punish a 5 ms error as harshly as a 200 ms one; smoothing
    encodes that near-misses are nearly right, which is what the tolerance metric says.
    """
    bins = displacement_bins()
    t = np.exp(-0.5 * ((bins - displacement_sec) / sigma_sec) ** 2)
    s = t.sum()
    return (t / s).astype(np.float32) if s > 0 else t.astype(np.float32)


def build_decoder_examples(
    clips: list[dict],
    decode_fn,
    min_iou: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Proposals taken from the decoder's real output, matched to references by IoU.

    Uniform jitter teaches the refiner to undo a symmetric, artificial displacement.  The
    errors it must actually fix are the decoder's, which are asymmetric and correlated
    with the shape of the activity curve - a slow fade-in biases the onset late, for
    instance.  Training on real proposals as well as jittered ones covers both.
    """
    windows, kinds, targets = [], [], []
    for c in clips:
        act, on, off = c["activity"], c["onset"], c["offset"]
        spans = decode_fn(c)
        if not spans:
            continue
        for ev in c["events"]:
            e_on, e_off = float(ev["onset"]), float(ev["offset"])
            best, best_iou = None, 0.0
            for s_on, s_off in spans:
                inter = max(0.0, min(e_off, s_off) - max(e_on, s_on))
                union = max(e_off, s_off) - min(e_on, s_on)
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best, best_iou = (s_on, s_off), iou
            if best is None or best_iou < min_iou:
                continue
            for kind, proposed, truth in ((0, best[0], e_on), (1, best[1], e_off)):
                disp = float(np.clip(truth - proposed, -MAX_DISP_SEC, MAX_DISP_SEC))
                windows.append(extract_window(act, on, off, int(round(proposed / FRAME_SEC))))
                kinds.append(kind)
                targets.append(soft_target(disp))
    if not windows:
        return (np.zeros((0, 3, 2 * HALF_WIN_FRAMES + 1), np.float32),
                np.zeros((0,), np.int64), np.zeros((0, N_BINS), np.float32))
    return np.stack(windows), np.array(kinds, dtype=np.int64), np.stack(targets)


def build_examples(
    clips: list[dict],
    rng: np.random.Generator,
    jitter_sec: float = 0.25,
    n_jitter: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Jittered proposals from reference endpoints.

    Each true endpoint yields several proposals displaced by a random amount; the target
    is the displacement that undoes the jitter.  Proposals are drawn uniformly rather than
    from the model's errors so the refiner sees the full correction range, including the
    large errors that currently fall outside the collar.
    """
    windows, kinds, targets = [], [], []
    for c in clips:
        act, on, off = c["activity"], c["onset"], c["offset"]
        n = act.shape[0]
        for ev in c["events"]:
            for kind, t_true in ((0, ev["onset"]), (1, ev["offset"])):
                for _ in range(n_jitter):
                    disp = float(rng.uniform(-jitter_sec, jitter_sec))
                    proposed = t_true + disp
                    idx = int(round(proposed / FRAME_SEC))
                    if not (-HALF_WIN_FRAMES <= idx <= n + HALF_WIN_FRAMES):
                        continue
                    windows.append(extract_window(act, on, off, idx))
                    kinds.append(kind)
                    # the correction that maps the proposal back onto the truth
                    targets.append(soft_target(float(np.clip(-disp, -MAX_DISP_SEC, MAX_DISP_SEC))))
    if not windows:
        return (np.zeros((0, 3, 2 * HALF_WIN_FRAMES + 1), np.float32),
                np.zeros((0,), np.int64), np.zeros((0, N_BINS), np.float32))
    return (np.stack(windows), np.array(kinds, dtype=np.int64), np.stack(targets))


def refine_spans(
    spans: list[tuple[float, float]],
    activity: np.ndarray,
    onset: np.ndarray,
    offset: np.ndarray,
    model: BoundaryRefiner,
    device: str,
    duration: float | None = None,
    max_shift: float = 0.25,
) -> list[tuple[float, float]]:
    if not spans:
        return spans
    windows, kinds = [], []
    for a, b in spans:
        windows.append(extract_window(activity, onset, offset, int(round(a / FRAME_SEC))))
        kinds.append(0)
        windows.append(extract_window(activity, onset, offset, int(round(b / FRAME_SEC))))
        kinds.append(1)
    w = torch.from_numpy(np.stack(windows)).to(device)
    k = torch.tensor(kinds, dtype=torch.long, device=device)
    shifts = model.expected_shift(w, k).cpu().numpy()

    out = []
    for i, (a, b) in enumerate(spans):
        na = a + float(np.clip(shifts[2 * i], -max_shift, max_shift))
        nb = b + float(np.clip(shifts[2 * i + 1], -max_shift, max_shift))
        if duration is not None:
            na, nb = max(0.0, min(na, duration)), max(0.0, min(nb, duration))
        out.append((round(na, 4), round(nb, 4)) if nb - na >= 1e-3 else (a, b))
    return out


def load_refiner(path: str, device) -> "BoundaryRefiner":
    """Load a refiner checkpoint; `hidden` is stored since 27 Sep (older files are 128)."""
    ck = torch.load(path, map_location="cpu")
    m = BoundaryRefiner(hidden=int(ck.get("hidden", 128))).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    return m
