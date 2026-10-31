"""Build a speaker-disjoint train/val split from the prepared manifest.

Design decisions, and why:

  Validation is drawn from Gold only.  Silver timestamps have unverified annotator
  agreement, so scoring against them measures agreement with noisy labels rather than
  with the reference the leaderboard uses.  Gold is 21.8 h, so a ~20% held-out slice
  lands near 4 h - the same size as the organisers' own (still unreleased) val split.

  The split is by speaker, not by clip.  Vaani records many clips per session, so a
  random clip split would leak the same speaker, room and microphone into both sides
  and inflate the score.  Every clip of a held-out speaker is removed from training
  across *all* tiers, not just Gold.

  All three tiers are kept for training.  Bronze clips carry noise categories but no
  timestamps, so their empty event list is missing data rather than silence; the dataset
  marks them has_frames=0 and they reach the model only through the clip-level MIL loss.
  Using every tier effectively is explicitly the point of this challenge.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load_manifest(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-hours", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    recs = load_manifest(args.manifest)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_tier = defaultdict(list)
    for r in recs:
        by_tier[r["tier"]].append(r)
    print("tier totals:")
    for tier, rs in sorted(by_tier.items()):
        hours = sum(r["duration"] for r in rs) / 3600
        print(f"  {tier:8s} {len(rs):7d} clips  {hours:7.2f} h")

    # candidate val speakers: those whose gold clips carry at least one event
    gold_by_speaker = defaultdict(list)
    for r in by_tier["gold"]:
        if r["events"]:
            gold_by_speaker[r["speaker"]].append(r)

    speakers = sorted(gold_by_speaker)
    rng = random.Random(args.seed)
    rng.shuffle(speakers)

    val_speakers: set[str] = set()
    val_seconds = 0.0
    target = args.val_hours * 3600
    for spk in speakers:
        if val_seconds >= target:
            break
        val_speakers.add(spk)
        val_seconds += sum(r["duration"] for r in gold_by_speaker[spk])

    val = [r for r in by_tier["gold"] if r["speaker"] in val_speakers and r["events"]]
    # all three tiers train; Bronze reaches the model only through the clip-level MIL
    # loss, which the dataset enforces via has_frames=0
    train = [r for r in recs if r["speaker"] not in val_speakers]

    # leakage assertions - cheap, and the whole point of the split
    train_speakers = {r["speaker"] for r in train}
    assert not (train_speakers & val_speakers), "speaker leak between train and val"
    val_ids = {r["clip_id"] for r in val}
    assert not (val_ids & {r["clip_id"] for r in train}), "clip leak between train and val"

    for name, rows in (("train", train), ("val", val)):
        path = out_dir / f"{name}.jsonl"
        with open(path, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        hours = sum(r["duration"] for r in rows) / 3600
        n_events = sum(len(r["events"]) for r in rows)
        tiers = defaultdict(int)
        for r in rows:
            tiers[r["tier"]] += 1
        print(
            f"\n{name}: {len(rows)} clips, {hours:.2f} h, {n_events} events, "
            f"{len({r['speaker'] for r in rows})} speakers, tiers={dict(tiers)}"
        )
        print(f"  -> {path}")

    # the val reference in scorer format, so the scorer never re-reads the manifest
    ref_path = out_dir / "val_ref.jsonl"
    with open(ref_path, "w") as fh:
        for r in val:
            fh.write(
                json.dumps(
                    {
                        "clip_id": r["clip_id"],
                        "duration": r["duration"],
                        "events": [
                            {"onset": e["onset"], "offset": e["offset"]} for e in r["events"]
                        ],
                    }
                )
                + "\n"
            )
    print(f"  -> {ref_path}")


if __name__ == "__main__":
    main()
