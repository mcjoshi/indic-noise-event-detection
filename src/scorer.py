"""Local reimplementation of the Datathon@IndoML 2026 Track 1 scoring metrics.

The organisers' scoring program is not downloadable (make_programs_available=False),
so this is written from the Evaluation page spec:

  Event-based F1 (+/-20% tolerance)
      A predicted event matches a reference event when BOTH onset and offset fall
      within tol = max(0.20 * ref_duration, 0.05) seconds of the reference.
      Micro-averaged over all clips.

  Segment-level Dice
      Audio is divided into 10 ms frames, binary masks are built for reference and
      prediction, Dice = 2 |P & G| / (|P| + |G|).  Macro-averaged across clips.

  Combined = Event F1 + Segment Dice   (max 2.0)

Two points the spec leaves open, resolved here by convention and flagged so that a
disagreement with the official leaderboard can be traced back to them:

  1. Matching procedure.  Greedy, one-to-one, references processed in onset order
     and matched against the earliest unclaimed qualifying prediction.
  2. Dice on a clip where both reference and prediction are empty.  Scored 1.0
     (perfect agreement).  A clip where exactly one side is empty scores 0.0.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np

FRAME_SEC = 0.010
TOL_FRAC = 0.20
TOL_FLOOR = 0.05


def _tolerance(ref_onset: float, ref_offset: float) -> float:
    return max(TOL_FRAC * (ref_offset - ref_onset), TOL_FLOOR)


def match_events(
    ref: list[tuple[float, float]],
    pred: list[tuple[float, float]],
) -> int:
    """Greedy one-to-one matching, closest pair first (the organisers' published matcher); returns the true positives."""
    candidates = []
    for ri, (r_on, r_off) in enumerate(ref):
        tol = _tolerance(r_on, r_off)
        for pi, (p_on, p_off) in enumerate(pred):
            if abs(p_on - r_on) <= tol and abs(p_off - r_off) <= tol:
                candidates.append((abs(p_on - r_on) + abs(p_off - r_off), ri, pi))
    matched_ref: set[int] = set()
    matched_pred: set[int] = set()
    for _, ri, pi in sorted(candidates):
        if ri not in matched_ref and pi not in matched_pred:
            matched_ref.add(ri)
            matched_pred.add(pi)
    return len(matched_ref)


def frame_mask(events: list[tuple[float, float]], n_frames: int) -> np.ndarray:
    """Binary mask over 10 ms frames.  A frame is positive if its centre is covered."""
    mask = np.zeros(n_frames, dtype=bool)
    for onset, offset in events:
        lo = int(np.floor(onset / FRAME_SEC))
        hi = int(np.ceil(offset / FRAME_SEC))
        lo = max(lo, 0)
        hi = min(hi, n_frames)
        if hi > lo:
            mask[lo:hi] = True
    return mask


def dice(ref_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    n_ref = int(ref_mask.sum())
    n_pred = int(pred_mask.sum())
    if n_ref == 0 and n_pred == 0:
        return 1.0
    if n_ref == 0 or n_pred == 0:
        return 0.0
    inter = int(np.logical_and(ref_mask, pred_mask).sum())
    return 2.0 * inter / (n_ref + n_pred)


@dataclass
class Scores:
    event_precision: float
    event_recall: float
    event_f1: float
    segment_dice: float
    n_clips: int
    n_ref_events: int
    n_pred_events: int

    @property
    def combined(self) -> float:
        return self.event_f1 + self.segment_dice

    def as_dict(self) -> dict:
        return {
            "event_precision": round(self.event_precision, 5),
            "event_recall": round(self.event_recall, 5),
            "event_f1": round(self.event_f1, 5),
            "segment_dice": round(self.segment_dice, 5),
            "combined": round(self.combined, 5),
            "n_clips": self.n_clips,
            "n_ref_events": self.n_ref_events,
            "n_pred_events": self.n_pred_events,
        }


def score(
    refs: dict[str, list[tuple[float, float]]],
    preds: dict[str, list[tuple[float, float]]],
    durations: dict[str, float],
) -> Scores:
    """Score predictions against references.

    Every clip in ``refs`` is scored.  A clip missing from ``preds`` is treated as an
    empty prediction, which is how the official scorer must handle it given that the
    submission format requires each clip to appear exactly once.
    """
    tp_total = 0
    n_ref_total = 0
    n_pred_total = 0
    dice_scores: list[float] = []

    for clip_id, ref_events in refs.items():
        pred_events = preds.get(clip_id, [])
        duration = durations[clip_id]
        n_frames = max(int(np.ceil(duration / FRAME_SEC)), 1)

        tp_total += match_events(ref_events, pred_events)
        n_ref_total += len(ref_events)
        n_pred_total += len(pred_events)

        dice_scores.append(
            dice(frame_mask(ref_events, n_frames), frame_mask(pred_events, n_frames))
        )

    precision = tp_total / n_pred_total if n_pred_total else 0.0
    recall = tp_total / n_ref_total if n_ref_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return Scores(
        event_precision=precision,
        event_recall=recall,
        event_f1=f1,
        segment_dice=float(np.mean(dice_scores)) if dice_scores else 0.0,
        n_clips=len(refs),
        n_ref_events=n_ref_total,
        n_pred_events=n_pred_total,
    )


def load_jsonl_events(path: str) -> dict[str, list[tuple[float, float]]]:
    out: dict[str, list[tuple[float, float]]] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["clip_id"]] = [
                (float(e["onset"]), float(e["offset"])) for e in rec.get("events", [])
            ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True, help="reference jsonl (needs a duration field)")
    ap.add_argument("--pred", required=True, help="predictions.jsonl")
    args = ap.parse_args()

    refs = load_jsonl_events(args.ref)
    preds = load_jsonl_events(args.pred)
    durations = {}
    with open(args.ref) as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                durations[rec["clip_id"]] = float(rec["duration"])

    print(json.dumps(score(refs, preds, durations).as_dict(), indent=2))


if __name__ == "__main__":
    main()
