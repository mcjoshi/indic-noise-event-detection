"""A second activity channel, so the model can express two simultaneous events.

Measured cause: 28.8% of natural references share their prediction with another reference,
64.3% of those reference pairs genuinely *overlap* (median gap -0.429 s), and corpus-wide
30.6% of event pairs overlap.  A single binary activity curve cannot represent two
simultaneous events - one contiguous run is its correct output - so ORACLE-SPLIT (+0.7190)
is unreachable by any decode rule, which is exactly what three attempts found.

96.5% of clips containing overlap are depth-2, so one extra channel covers nearly all of it
and a DETR-style query head is not needed.

Targets come from greedily two-colouring each clip's events by onset: an event goes to
layer 1 unless it overlaps the last event already there, in which case it goes to layer 2.
Channel 1 keeps its existing meaning (union of all events) so the shipped head, the refiner
and the whole decode stack are untouched; channel 2 carries the *full extent* of the layer-2
events, not merely the overlapping region, so a decoded channel-2 span is a complete second
event that can match a reference within tolerance.

Bronze clips are excluded - their empty event list is missing data, not silence, so they
cannot supervise an overlap target any more than they can a frame target.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

FRAME_SEC = 0.01


def two_colour(events: list[dict]) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Split a clip's events into two layers, neither containing an internal overlap.

    Greedy by onset: this is optimal for interval-graph colouring on an interval set, and
    with 96.5% of overlapping clips at depth 2 the rare depth-3 clip simply loses its third
    event from supervision rather than corrupting a layer.
    """
    evs = sorted(((float(e["onset"]), float(e["offset"])) for e in events),
                 key=lambda t: (t[0], t[1]))
    l1: list[tuple[float, float]] = []
    l2: list[tuple[float, float]] = []
    for on, off in evs:
        if not l1 or on >= l1[-1][1]:
            l1.append((on, off))
        elif not l2 or on >= l2[-1][1]:
            l2.append((on, off))
        # depth >= 3 events are dropped: no third channel to hold them
    return l1, l2


def layer_target(spans: list[tuple[float, float]], n_frames: int) -> np.ndarray:
    y = np.zeros(n_frames, dtype=np.float32)
    for on, off in spans:
        a = max(0, int(round(on / FRAME_SEC)))
        b = min(n_frames, max(int(round(off / FRAME_SEC)), a + 1))
        if b > a:
            y[a:b] = 1.0
    return y


class OverlapHead(nn.Module):
    """Small temporal head over frozen encoder states, predicting the layer-2 activity.

    Deliberately small and convolutional rather than another recurrence: it is trained on
    frozen v11 states, so its only job is to read a second event out of a representation
    that already encodes the acoustics.  Keeping it cheap is what makes the frozen-encoder
    derisking loop take minutes instead of hours.
    """

    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden, 5, padding=2), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, 5, padding=2, dilation=1), nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2), nn.GELU(),
        )
        self.out = nn.Conv1d(hidden, 1, 1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, T) logits."""
        x = states.transpose(1, 2)
        return self.out(self.net(x)).squeeze(1)
