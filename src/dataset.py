"""Dataset and frame-target construction for frame-level noise event detection.

The frame grid is 10 ms - identical to the grid the segment-Dice metric uses.  The
tolerance at the median event is only ~90 ms, so any coarser grid would quantise
boundaries at a scale comparable to the entire error budget.
"""

from __future__ import annotations
from paths import DATA, PRETRAINED, WORK  # noqa: F401

import json
import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from epn_head import epn_targets

SR = 16000
HOP = 160          # 10 ms

DATA2 = DATA


def rel_wav(wav: str) -> Path:
    """Manifest wav paths are relative to the audio root, except synthetic mixtures, which
    are absolute under DATA2; both map to one relative key for sibling directories."""
    p = Path(wav)
    return p.relative_to(DATA2) if p.is_absolute() else p
FRAME_SEC = HOP / SR

CATEGORIES = [
    "animal",
    "vehicle_traffic",
    "baby_child",
    "singing_music",
    "phone_signal_alarm",
    "appliance_machine",
    "human_non_speech",
]
CATEGORY_ALIASES = {
    "animal_sound": "animal",
    "baby_child_noise": "baby_child",
    "phone_signal": "phone_signal_alarm",
    "alarm": "phone_signal_alarm",
    "appliance": "appliance_machine",
    "machine": "appliance_machine",
}
CAT_INDEX = {c: i for i, c in enumerate(CATEGORIES)}
TIER_ID = {"gold": 0, "silver": 1, "bronze": 2, "bronze_pseudo": 3, "parent_pseudo": 4, "parent": 5, "synth": 6}


def canonical_category(name: str) -> str | None:
    name = (name or "").strip().lower()
    name = CATEGORY_ALIASES.get(name, name)
    return name if name in CAT_INDEX else None


def n_frames_for(n_samples: int) -> int:
    return max(int(round(n_samples / HOP)), 1)


def blur(x: np.ndarray, sigma_frames: float) -> np.ndarray:
    """Small Gaussian smoothing used to soften Silver's unverified boundaries."""
    if sigma_frames <= 0:
        return x
    radius = int(max(1, round(3 * sigma_frames)))
    t = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (t / sigma_frames) ** 2)
    kernel /= kernel.sum()
    return np.convolve(x, kernel, mode="same").astype(np.float32)


def boundary_targets(
    events: list[dict],
    n_frames: int,
    offset_sec: float = 0.0,
    sigma_frames: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian bumps at event onsets and offsets, as separate targets.

    Event-F1 is decided entirely by boundary placement inside a ~90 ms collar, but a
    frame-activity target only says "event here", never "the event *starts* here".  Two
    dedicated heads give the model a gradient aimed at the quantity actually scored, and
    `sigma_frames` is where the annotation tier enters: Gold boundaries are trusted
    tightly, Silver's unverified ones get a wide bump, which encodes them as uncertain
    observations rather than exact truth.
    """
    onset = np.zeros(n_frames, dtype=np.float32)
    offset = np.zeros(n_frames, dtype=np.float32)
    if n_frames == 0:
        return onset, offset
    grid = np.arange(n_frames, dtype=np.float32)
    for ev in events:
        for t, target in ((float(ev["onset"]), onset), (float(ev["offset"]), offset)):
            pos = (t - offset_sec) / FRAME_SEC
            if -3 * sigma_frames <= pos <= n_frames + 3 * sigma_frames:
                bump = np.exp(-0.5 * ((grid - pos) / sigma_frames) ** 2)
                np.maximum(target, bump, out=target)
    return onset, offset


MEDIAN_EVENT_SEC = 0.448   # measured on the Gold validation split


def event_weights(
    events: list[dict],
    n_frames: int,
    offset_sec: float = 0.0,
    lo: float = 0.25,
    hi: float = 8.0,
) -> np.ndarray:
    """Per-frame weights that make every event contribute equally, whatever its length.

    A frame-level BCE weights an event by its duration, so a 3 s event supplies ~30x the
    gradient of a 0.1 s one.  That is the direct cause of the measured failure: recall is
    0.118 for events under 250 ms against ~0.39 elsewhere.  The survey put it plainly -
    "frame weighting is not event weighting" - and the fix is to normalise each event's
    contribution by its duration (Schmid et al., Boundary-Aware Optimization, 2026).

    Each event's frames are scaled by median_duration / event_duration, clamped so a very
    short event cannot dominate a batch and a very long one is not silenced.  Background
    frames keep weight 1.
    """
    w = np.ones(n_frames, dtype=np.float32)
    if n_frames == 0:
        return w
    for ev in events:
        onset, offset = float(ev["onset"]), float(ev["offset"])
        dur = max(offset - onset, 1e-3)
        scale = float(np.clip(MEDIAN_EVENT_SEC / dur, lo, hi))
        a = int(np.floor((onset - offset_sec) / FRAME_SEC))
        b = int(np.ceil((offset - offset_sec) / FRAME_SEC))
        a, b = max(a, 0), min(b, n_frames)
        if b > a:
            np.maximum(w[a:b], scale, out=w[a:b])
    return w


def boundary_weight(
    events: list[dict],
    n_frames: int,
    offset_sec: float = 0.0,
    strength: float = 2.0,
    sigma_frames: float = 5.0,
) -> np.ndarray:
    """Up-weight frames near an event boundary in the PRIMARY frame loss.

    Distinct from the dedicated onset/offset heads: those add extra outputs, whereas this
    reshapes the activity loss itself so that the frames deciding whether a prediction
    lands inside the collar carry more gradient than the middle of a long event.  Reported
    at +6.43% relative event-F1 over plain BCE on a CRNN (arXiv:2403.13254).

    Weight is 1 + strength * exp(-d^2 / 2 sigma^2) with d the distance in frames to the
    nearest boundary, so the interior of an event and pure background keep weight 1.
    """
    w = np.ones(n_frames, dtype=np.float32)
    if n_frames == 0 or not events:
        return w
    grid = np.arange(n_frames, dtype=np.float32)
    dist = np.full(n_frames, np.inf, dtype=np.float32)
    for ev in events:
        for t in (float(ev["onset"]), float(ev["offset"])):
            pos = (t - offset_sec) / FRAME_SEC
            np.minimum(dist, np.abs(grid - pos), out=dist)
    return (1.0 + strength * np.exp(-0.5 * (dist / sigma_frames) ** 2)).astype(np.float32)


def layer2_target(events, n_frames: int, offset_sec: float = 0.0) -> np.ndarray:
    """Frames where a *second*, overlapping event is active.

    Events are greedily two-coloured by onset: an event joins layer 1 unless it overlaps
    the last one already there, in which case it joins layer 2.  Layer 2 carries the full
    extent of the overlapping event, not merely the overlapping region, so a decoded
    layer-2 span is a complete event that can match a reference inside the tolerance.

    Depth-3 events are dropped - there is no third channel - which costs 1.6% of clips.
    """
    from overlap_head import two_colour

    _, l2 = two_colour(events)
    y = np.zeros(n_frames, dtype=np.float32)
    for on, off in l2:
        a = int(round((on - offset_sec) / FRAME_SEC))
        b = int(round((off - offset_sec) / FRAME_SEC))
        a, b = max(0, a), min(n_frames, b)
        if b > a:
            y[a:b] = 1.0
    return y


def frame_targets(
    events: list[dict],
    n_frames: int,
    offset_sec: float = 0.0,
    blur_sigma: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Soft frame targets: the fraction of each 10 ms frame covered by an event."""
    starts = np.arange(n_frames, dtype=np.float64) * FRAME_SEC + offset_sec
    ends = starts + FRAME_SEC

    binary = np.zeros(n_frames, dtype=np.float32)
    per_cat = np.zeros((n_frames, len(CATEGORIES)), dtype=np.float32)

    for ev in events:
        onset, offset = float(ev["onset"]), float(ev["offset"])
        overlap = np.clip(np.minimum(ends, offset) - np.maximum(starts, onset), 0.0, FRAME_SEC)
        frac = (overlap / FRAME_SEC).astype(np.float32)
        np.maximum(binary, frac, out=binary)
        cat = canonical_category(ev.get("category", ""))
        if cat is not None:
            idx = CAT_INDEX[cat]
            np.maximum(per_cat[:, idx], frac, out=per_cat[:, idx])

    if blur_sigma > 0:
        binary = np.clip(blur(binary, blur_sigma), 0.0, 1.0)
        for c in range(per_cat.shape[1]):
            if per_cat[:, c].any():
                per_cat[:, c] = np.clip(blur(per_cat[:, c], blur_sigma), 0.0, 1.0)

    return binary, per_cat


class VaaniFrameDataset(Dataset):
    """Clips with soft frame targets plus clip-level targets.

    Bronze clips carry no timestamps, so they are marked `has_frames=0` and only ever
    contribute through the clip-level MIL loss.  Their empty event list is missing data,
    not silence, and must never be used as a negative frame target.
    """

    def __init__(
        self,
        jsonl_path: str,
        audio_root: str,
        max_sec: float | None = 10.0,
        train: bool = True,
        silver_blur: float = 1.5,
        gold_sigma: float = 2.0,
        silver_sigma: float = 7.0,
        bnd_weight_strength: float = 0.0,
        soft_weight: float = 0.0,
        seed: int = 0,
        residual_root: str | None = None,
        gain_aug_db: float = 0.0,
        speed_aug: str = "",
        gold_oversample: int = 1,
    ):
        if gold_oversample < 1:
            raise ValueError("gold_oversample must be at least 1")
        self.residual_root = Path(residual_root) if residual_root else None
        self.gain_aug_db = gain_aug_db
        self.speed_aug = [float(x) for x in speed_aug.split(",") if x.strip()] if speed_aug else []
        self.bnd_weight_strength = bnd_weight_strength
        self.soft_weight = soft_weight
        self.gold_sigma = gold_sigma
        self.silver_sigma = silver_sigma
        self.records = [json.loads(l) for l in open(jsonl_path) if l.strip()]
        if train and gold_oversample > 1:
            gold_natural = [
                rec for rec in self.records
                if rec["tier"] == "gold"
                and not rec.get("synthetic", False)
                and "synth" not in rec["clip_id"].lower()
            ]
            self.records.extend(gold_natural * (gold_oversample - 1))
        self.audio_root = Path(audio_root)
        self.max_samples = int(max_sec * SR) if max_sec else None
        self.train = train
        self.silver_blur = silver_blur
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._rng_owner = None

    def __len__(self) -> int:
        return len(self.records)

    def _worker_rng(self) -> np.random.Generator:
        # a Generator built in __init__ is copied into every DataLoader worker, so all workers
        # would draw the same crops and gains; reseed once per worker from torch's per-worker seed
        info = torch.utils.data.get_worker_info()
        owner = None if info is None else (info.id, torch.initial_seed())
        if owner != self._rng_owner:
            self.rng = np.random.default_rng([self.seed, owner[0], owner[1] % 2**32] if owner else self.seed)
            self._rng_owner = owner
        return self.rng

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        wav, sr = sf.read(self.audio_root / rec["wav"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        assert sr == SR, f"unexpected sample rate {sr} in {rec['clip_id']}"

        tier = rec["tier"]
        res = None
        if self.residual_root is not None:
            res, _ = sf.read(self.residual_root / rel_wav(rec["wav"]), dtype="float32")
            if res.ndim > 1:
                res = res.mean(axis=1)
            res = res[: len(wav)]
        if self.train and self.speed_aug:
            # tape-speed perturbation: resample the clip and scale the event times with it, so the
            # model sees each event at several durations. The only augmentation axis never used here.
            sp = float(self._worker_rng().choice(self.speed_aug))
            if sp != 1.0:
                from fractions import Fraction
                from scipy.signal import resample_poly
                fr = Fraction(1.0 / sp).limit_denominator(100)
                wav = resample_poly(wav, fr.numerator, fr.denominator).astype(np.float32)
                if res is not None:
                    res = resample_poly(res, fr.numerator, fr.denominator).astype(np.float32)[: len(wav)]
                rec = {**rec, "events": [{**e, "onset": e["onset"] / sp, "offset": e["offset"] / sp}
                                         for e in rec["events"]]}

        offset_sec = 0.0
        if self.max_samples and len(wav) > self.max_samples:
            start = (
                int(self._worker_rng().integers(0, len(wav) - self.max_samples + 1))
                if self.train
                else 0
            )
            wav = wav[start : start + self.max_samples]
            if res is not None:
                res = res[start : start + self.max_samples]
            offset_sec = start / SR
        if self.train and self.gain_aug_db > 0:
            # per-clip level augmentation: nothing in the pipeline normalises level and the eval
            # synthetic clips sit 13-19 dB from our synthetic training clips
            g = float(10 ** (self._worker_rng().uniform(-self.gain_aug_db, self.gain_aug_db) / 20))
            wav = wav * g
            if res is not None:
                res = res * g

        n_fr = n_frames_for(len(wav))
        # bronze_pseudo boundaries come from a teacher constrained by the human
        # category label, so they are treated as less certain than Silver's
        blur_sigma = 0.0
        if self.train and tier == "silver":
            blur_sigma = self.silver_blur
        elif self.train and tier == "bronze_pseudo":
            blur_sigma = self.silver_blur * 1.5
        binary, per_cat = frame_targets(rec["events"], n_fr, offset_sec, blur_sigma)
        if self.train and self.soft_weight > 0 and rec.get("soft"):
            # ensemble vote fraction per frame (knowledge distillation target), mixed with the hard label
            soft = np.zeros(n_fr, dtype=np.float32)
            for a, b, v in rec["soft"]:
                i0 = max(0, int(round((a - offset_sec) / FRAME_SEC))); i1 = min(n_fr, int(round((b - offset_sec) / FRAME_SEC)))
                if i1 > i0:
                    soft[i0:i1] = np.maximum(soft[i0:i1], v)
            binary = ((1 - self.soft_weight) * binary + self.soft_weight * soft).astype(binary.dtype)
        # Silver's boundaries are unverified, so they are supervised as a wide Gaussian
        # (70 ms) rather than a tight one (20 ms): the model is told roughly where the
        # boundary is without being penalised for disagreeing with an unreliable label.
        b_sigma = self.gold_sigma
        if tier == "silver":
            b_sigma = self.silver_sigma
        elif tier == "bronze_pseudo":
            b_sigma = self.silver_sigma * 1.5
        onset_t, offset_t = boundary_targets(rec["events"], n_fr, offset_sec, b_sigma)
        ev_w = event_weights(rec["events"], n_fr, offset_sec)
        l2_t = layer2_target(rec["events"], n_fr, offset_sec)
        epn_t, epn_m = epn_targets(rec["events"], n_fr, offset_sec)
        clip_end = offset_sec + len(wav) / SR
        collar_events = [
            (float(ev["onset"]) - offset_sec, float(ev["offset"]) - offset_sec)
            for ev in rec["events"]
            if float(ev["onset"]) >= offset_sec and float(ev["offset"]) <= clip_end
        ]
        if self.bnd_weight_strength > 0:
            ev_w = ev_w * boundary_weight(rec["events"], n_fr, offset_sec,
                                          self.bnd_weight_strength)

        # clip-level targets come from the category list, which every tier has
        clip_cat = np.zeros(len(CATEGORIES), dtype=np.float32)
        for c in rec.get("categories") or []:
            cc = canonical_category(c)
            if cc is not None:
                clip_cat[CAT_INDEX[cc]] = 1.0

        return {
            "clip_id": rec["clip_id"],
            "wav": torch.from_numpy(wav),
            "res": torch.from_numpy(res) if res is not None else None,
            "target": torch.from_numpy(binary),
            "target_cat": torch.from_numpy(per_cat),
            "event_weight": torch.from_numpy(ev_w),
            "target_onset": torch.from_numpy(onset_t),
            "target_offset": torch.from_numpy(offset_t),
            "target_l2": torch.from_numpy(l2_t),
            "target_epn": torch.from_numpy(epn_t),
            "mask_epn": torch.from_numpy(epn_m),
            "collar_events": collar_events,
            "clip_cat": torch.from_numpy(clip_cat),
            # every clip in this corpus carries at least one noise category, so the
            # clip-level activity target is 1 everywhere; kept explicit in case a
            # future release adds true negatives
            "clip_act": float(len(rec.get("categories") or []) > 0),
            # parent-corpus clips carry no human category tags: their clip/aux losses are masked
            "has_cats": float(rec.get("categories") is not None),
            "has_frames": float(rec.get("has_frames", tier not in ("bronze", "parent"))),
            "bnd_rel": float(rec.get("bnd_rel", 1.0)),
            "tier": TIER_ID[tier],
            "duration": rec["duration"],
        }


class VaaniWaveDataset(Dataset):
    """Waveform-only dataset for objectives that must not consume event labels."""

    def __init__(self, jsonl_path: str, audio_root: str, crop_sec: float = 8.0,
                 train: bool = True, seed: int = 0, holdout_percent: float = 0.0,
                 holdout: bool = False):
        if not 0 <= holdout_percent < 100:
            raise ValueError("holdout_percent must be in [0, 100)")
        self.audio_root = Path(audio_root)
        self.crop_samples = int(crop_sec * SR)
        self.train = train
        self.seed = seed
        self.epoch = 0
        self.records = []
        for line in open(jsonl_path):
            if not line.strip():
                continue
            rec = json.loads(line)
            bucket = int.from_bytes(
                hashlib.sha256(rec["clip_id"].encode()).digest()[:8], "big"
            ) % 100
            if (bucket < holdout_percent) == holdout:
                self.records.append(rec)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def _path(self, wav: str) -> Path:
        path = Path(wav)
        if path.is_absolute():
            if not path.exists():
                raise FileNotFoundError(f"absolute waveform path does not exist: {path}")
            return path
        return self.audio_root / rel_wav(wav)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        wav, sr = sf.read(self._path(rec["wav"]), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SR:
            raise ValueError(f"unexpected sample rate {sr} in {rec['clip_id']}")
        true_samples = len(wav)
        if true_samples > self.crop_samples:
            if self.train:
                seed = (self.seed * 1_000_003 + self.epoch * 9_176 + idx) & 0xFFFFFFFF
                start = int(np.random.default_rng(seed).integers(0, true_samples - self.crop_samples + 1))
            else:
                start = 0
            wav = wav[start:start + self.crop_samples]
            true_samples = self.crop_samples
        padded = np.zeros(self.crop_samples, dtype=np.float32)
        padded[:true_samples] = wav
        padding_mask = np.ones(self.crop_samples, dtype=np.bool_)
        padding_mask[:true_samples] = False
        return {
            "clip_id": rec["clip_id"],
            "wav": torch.from_numpy(padded),
            "padding_mask": torch.from_numpy(padding_mask),
        }


def collate(batch: list[dict]) -> dict:
    max_samples = max(b["wav"].shape[0] for b in batch)
    max_frames = max(b["target"].shape[0] for b in batch)
    bs = len(batch)

    wav = torch.zeros(bs, max_samples)
    target = torch.zeros(bs, max_frames)
    target_cat = torch.zeros(bs, max_frames, len(CATEGORIES))
    event_weight = torch.ones(bs, max_frames)
    target_onset = torch.zeros(bs, max_frames)
    target_l2 = torch.zeros(bs, max_frames)
    target_offset = torch.zeros(bs, max_frames)
    frame_mask = torch.zeros(bs, max_frames)
    target_epn = torch.zeros(bs, max_frames, 4)
    mask_epn = torch.zeros(bs, max_frames, 2)

    res = torch.zeros(bs, max_samples) if batch[0].get("res") is not None else None
    for i, b in enumerate(batch):
        n_s, n_f = b["wav"].shape[0], b["target"].shape[0]
        wav[i, :n_s] = b["wav"]
        if res is not None:
            res[i, :b["res"].shape[0]] = b["res"]
        target[i, :n_f] = b["target"]
        target_cat[i, :n_f] = b["target_cat"]
        event_weight[i, :n_f] = b["event_weight"]
        target_onset[i, :n_f] = b["target_onset"]
        target_l2[i, :n_f] = b["target_l2"]
        target_offset[i, :n_f] = b["target_offset"]
        target_epn[i, :n_f] = b["target_epn"]
        mask_epn[i, :n_f] = b["mask_epn"]
        frame_mask[i, :n_f] = 1.0

    return {
        "clip_ids": [b["clip_id"] for b in batch],
        "wav": wav,
        "res": res,
        "target": target,
        "target_cat": target_cat,
        "event_weight": event_weight,
        "target_onset": target_onset,
        "target_l2": target_l2,
        "target_offset": target_offset,
        "target_epn": target_epn,
        "mask_epn": mask_epn,
        "collar_events": [b["collar_events"] for b in batch],
        "collar_mask": torch.ones(bs),
        "frame_mask": frame_mask,
        "clip_cat": torch.stack([b["clip_cat"] for b in batch]),
        "clip_act": torch.tensor([b["clip_act"] for b in batch]),
        "has_frames": torch.tensor([b["has_frames"] for b in batch]),
        "bnd_rel": torch.tensor([b["bnd_rel"] for b in batch]),
        "has_cats": torch.tensor([b["has_cats"] for b in batch]),
        "tier": torch.tensor([b["tier"] for b in batch], dtype=torch.long),
        "durations": [b["duration"] for b in batch],
    }
