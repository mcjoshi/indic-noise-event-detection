"""Turn the downloaded Vaani parquet shards into wav files plus a flat manifest.

The parquet shards embed the audio bytes, which makes random access during training
slow (182 shards, 90k clips).  This writes each clip out once as a 16 kHz mono wav and
emits a single manifest.jsonl that the dataset class and the baselines both read.

Clip filenames look like

    IISc_VaaniProject_K_ArunachalPradesh_Longding_Longding844425030212709_049_GENERIC_0760_14530_15948.wav
                        ^state          ^district ^district+speaker id

so the speaker id can be recovered and used for a speaker-disjoint validation split,
which matches how the organisers built their own held-out set.

Usage:
    python3 src/prep_data.py --shards data/hf_train/data --out data/prepared
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

TIER_MAP = {
    "verified_timestamps": "gold",
    "unverified_timestamps": "silver",
    "no_timestamps": "bronze",
}

# Clip ids follow IISc_VaaniProject_<X>_<State>_<District>_<SESSION>_..., but the
# SESSION token itself has several shapes across collection drives:
#   Hyderabad844425030208139   (district + device id)
#   Ambi15455                  (locality + id)
#   Gang28299fcfbbd9           (locality + hex id)
# All three identify one recording session/speaker, so the whole token is the grouping
# key.  It is finer than district and never splits a speaker across folds, which is what
# a leakage-free validation split needs.
SESSION_FIELD_INDEX = 5


def parse_speaker(clip_id: str, district: str = "") -> str:
    parts = clip_id.split("_")
    if district:
        for i, p in enumerate(parts[:-1]):
            if p == district:
                return parts[i + 1]
    if len(parts) > SESSION_FIELD_INDEX:
        return parts[SESSION_FIELD_INDEX]
    return "unknown"


def _as_list(value) -> list[str]:
    """Parquet list columns come back as numpy arrays, which are not truthy-testable."""
    if value is None:
        return []
    return [str(v) for v in value]


def clean_events(raw, duration: float) -> list[dict]:
    """Parse, clip to the clip bounds, drop degenerate spans, sort by onset.

    `start`/`end` arrive as strings in the main release and as floats in the sample
    mirror, so both are handled.  Events are *not* merged here - overlap between
    different categories is real and the merging policy belongs to post-processing.
    """
    events = []
    for e in raw if raw is not None else []:
        try:
            onset = float(e["start"])
            offset = float(e["end"])
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(onset) and np.isfinite(offset)):
            continue
        onset = max(0.0, min(onset, duration))
        offset = max(0.0, min(offset, duration))
        if offset - onset < 1e-3:
            continue
        events.append(
            {
                "onset": round(onset, 4),
                "offset": round(offset, 4),
                "category": str(e.get("category", "")),
                "tag": str(e.get("tag", "")),
            }
        )
    events.sort(key=lambda x: (x["onset"], x["offset"]))
    return events


def process_shard(job: tuple[str, str, bool]) -> tuple[str, list[dict], dict]:
    shard_path, out_root, write_wav = job
    shard_name = Path(shard_path).stem
    wav_dir = Path(out_root) / "wav" / shard_name
    if write_wav:
        wav_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(shard_path)
    records: list[dict] = []
    stats = {"rows": 0, "skipped_audio": 0, "sr_mismatch": 0, "dur_mismatch": 0}

    for _, row in df.iterrows():
        stats["rows"] += 1
        audio = row["audio"]
        if not audio or not audio.get("bytes"):
            stats["skipped_audio"] += 1
            continue

        clip_id = Path(str(audio["path"])).stem
        wav_path = wav_dir / f"{clip_id}.wav"

        try:
            samples, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
        except Exception:
            stats["skipped_audio"] += 1
            continue

        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if sr != 16000:
            stats["sr_mismatch"] += 1

        # trust the decoded audio over the metadata field for the frame grid
        duration = len(samples) / sr
        if abs(duration - float(row["duration"])) > 0.05:
            stats["dur_mismatch"] += 1

        if write_wav and not wav_path.exists():
            sf.write(wav_path, samples, sr, subtype="PCM_16")

        quality = row.get("annotationQuality", "verified_timestamps")
        records.append(
            {
                "clip_id": clip_id,
                "wav": str(wav_path.relative_to(out_root)),
                "duration": round(duration, 4),
                "sr": int(sr),
                "tier": TIER_MAP.get(str(quality), str(quality)),
                "state": str(row.get("state", "")),
                "district": str(row.get("district", "")),
                "language": str(row.get("language", "")),
                "speaker": parse_speaker(clip_id, str(row.get("district", ""))),
                "categories": _as_list(row.get("NoiseCategory")),
                "transcript": str(row.get("transcript", "")),
                "events": clean_events(row.get("NoiseSubCategoryTimeStamp"), duration),
            }
        )

    return shard_name, records, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit-shards", type=int, default=0, help="0 = all")
    ap.add_argument("--no-wav", action="store_true", help="manifest only, skip wav extraction")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    shards = sorted(Path(args.shards).glob("*.parquet"))
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    if not shards:
        sys.exit(f"no parquet shards under {args.shards}")
    print(f"{len(shards)} shards -> {out_root}", flush=True)

    jobs = [(str(p), str(out_root), not args.no_wav) for p in shards]
    total = {"rows": 0, "skipped_audio": 0, "sr_mismatch": 0, "dur_mismatch": 0}
    n_written = 0

    manifest_path = out_root / "manifest.jsonl"
    with open(manifest_path, "w") as fh:
        with mp.Pool(args.workers) as pool:
            for i, (shard_name, records, stats) in enumerate(
                pool.imap_unordered(process_shard, jobs), 1
            ):
                for rec in records:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += len(records)
                for k in total:
                    total[k] += stats[k]
                print(
                    f"[{i}/{len(shards)}] {shard_name}: {len(records)} clips "
                    f"(total {n_written})",
                    flush=True,
                )

    print(f"\nwrote {n_written} records to {manifest_path}")
    print("stats:", json.dumps(total))
    if total["sr_mismatch"]:
        print(f"WARNING: {total['sr_mismatch']} clips were not 16 kHz")
    if total["dur_mismatch"]:
        print(f"WARNING: {total['dur_mismatch']} clips disagree with the duration field by >50 ms")


if __name__ == "__main__":
    main()
