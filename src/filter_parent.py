"""Turn the extracted parent-corpus manifest into an unlabeled pool: tier `parent`, no
category tags, and every clip id that appears in the competition train/val/test sets removed."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from paths import DATA


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--exclude-manifests", nargs="*", default=[f"{DATA}/splits/train_v7.jsonl", f"{DATA}/splits/val_mixed.jsonl"])
    ap.add_argument("--exclude-audio-dir", default=f"{DATA}/eval/extracted/audio")
    ap.add_argument("--min-sec", type=float, default=1.0)
    ap.add_argument("--max-sec", type=float, default=20.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    excl = set()
    for m in args.exclude_manifests:
        excl |= {json.loads(l)["clip_id"] for l in open(m) if l.strip()}
    test_dir = Path(args.exclude_audio_dir)
    if not test_dir.is_dir():
        sys.exit(f"--exclude-audio-dir {test_dir} not found: point it at the test audio so test clips are excluded")
    excl |= {p.stem for p in test_dir.rglob("*.wav")}
    n_in = n_out = n_ex = 0
    with open(args.out, "w") as fh:
        for l in open(args.manifest):
            if not l.strip():
                continue
            r = json.loads(l); n_in += 1
            if r["clip_id"] in excl:
                n_ex += 1; continue
            if not (args.min_sec <= r["duration"] <= args.max_sec):
                continue
            r = dict(r, tier="parent", categories=None, events=[])
            fh.write(json.dumps(r, ensure_ascii=False) + "\n"); n_out += 1
    print(f"{n_in} parent clips -> {n_out} kept ({n_ex} excluded as competition clips)")


if __name__ == "__main__":
    main()
