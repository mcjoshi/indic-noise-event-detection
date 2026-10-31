"""Silence gate for the synthetic route: drop a predicted span whose mean 10 ms frame level sits more than
--margin dB below the clip's own 95th-percentile frame level (offval_B2: the organisers' events never sit that
low, the pair's false alarms in near-silence do). Applied only to clips the condition detector routes as
synthetic; --check scores it on the official clips from the probe span cache."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paths import DATA  # noqa: E402
SR, FRAME = 16000, 0.010


def frame_db(path):
    x, _ = sf.read(path, dtype="float64")
    if x.ndim > 1:
        x = x.mean(1)
    n = int(SR * FRAME)
    k = max(1, len(x) // n)
    return 10.0 * np.log10((x[: k * n].reshape(k, n) ** 2).mean(1) + 1e-12)


def keep(f, ref, on, off, margin):
    lo = max(0, int(on / FRAME))
    hi = min(len(f), max(lo + 1, int(np.ceil(off / FRAME))))
    seg = f[lo:hi]
    return 10 * np.log10(np.mean(10 ** (seg / 10)) + 1e-12) >= ref - margin


def gate(spans, path, margin):
    f = frame_db(path)
    ref = float(np.percentile(f, 95))
    return [(a, b) for a, b in spans if keep(f, ref, a, b, margin)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred"); ap.add_argument("--out"); ap.add_argument("--audio-dir")
    ap.add_argument("--detector", default=str(ROOT / "work/condition_oracle_peak.json"))
    ap.add_argument("--margin", type=float, default=30.0)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        sys.path.insert(0, str(ROOT / "src"))
        from scorer import score
        blob = pickle.load(open(ROOT / "work/probe_off_synerr_spans_all.pkl", "rb"))
        wav = {json.loads(l)["clip_id"]: json.loads(l)["wav"] for l in open(DATA / "splits/official_val_syn.jsonl")}
        for tag, sp in blob["spans"].items():
            ids = list(sp)
            g = {c: gate(sp[c], DATA / "prepared" / wav[c], args.margin) for c in ids}
            s0 = score({c: blob["refs"][c] for c in ids}, sp, {c: blob["durs"][c] for c in ids})
            s1 = score({c: blob["refs"][c] for c in ids}, g, {c: blob["durs"][c] for c in ids})
            dropped = sum(len(sp[c]) - len(g[c]) for c in ids)
            print(f"CHECK {tag}: F1 {s0.event_f1:.4f}->{s1.event_f1:.4f} Dice {s0.segment_dice:.4f}->{s1.segment_dice:.4f} "
                  f"board-weighted {0.5624*(s1.event_f1-s0.event_f1)+0.3891*(s1.segment_dice-s0.segment_dice):+.4f} dropped {dropped}")
        return
    route = json.load(open(args.detector))["test_assignments"]
    n_in = n_out = touched = 0
    with open(args.out, "w") as fh:
        for l in open(args.pred):
            r = json.loads(l)
            ev = [(float(e["onset"]), float(e["offset"])) for e in r["events"]]
            if route.get(r["clip_id"]) == 1 and ev:
                kept = gate(ev, Path(args.audio_dir) / f"{r['clip_id']}.wav", args.margin)
                n_in += len(ev); n_out += len(kept); touched += len(kept) != len(ev)
                r["events"] = [{"onset": a, "offset": b} for a, b in kept]
            fh.write(json.dumps(r) + "\n")
    print(f"GATE margin {args.margin}: synthetic-route events {n_in} -> {n_out} ({n_in - n_out} dropped over {touched} clips) -> {args.out}")


if __name__ == "__main__":
    main()
