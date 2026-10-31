"""Build split-disjoint, exact-label synthetic Vaani mixtures on CPU.

Training uses train Gold beds, train Gold event crops, and ESC-50 folds 1-4.
Validation uses only the corresponding validation sources and ESC-50 fold 5.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import multiprocessing as mp
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from paths import DATA

from build_synth_mixtures import ESC_TO_VAANI, energetic_span, free_regions
from synthetic_aug import make_synthetic

SR = 16000
PRIORS = {
    "human_non_speech": 0.47, "vehicle_traffic": 0.23, "animal": 0.18,
    "baby_child": 0.08, "phone_signal_alarm": 0.03, "singing_music": 0.01,
    "appliance_machine": 0.005,
}


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)) + 1e-12)


def resample(x: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == SR:
        return x.astype(np.float32, copy=False)
    n = int(round(len(x) * SR / src_sr))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def resize(x: np.ndarray, n_samples: int) -> np.ndarray:
    if len(x) == n_samples:
        return x
    return np.interp(
        np.linspace(0, len(x) - 1, n_samples), np.arange(len(x)), x
    ).astype(np.float32)


def load_esc(esc_dir: str, allowed_folds: set[int]) -> list[dict]:
    frames = [pd.read_parquet(p) for p in sorted(glob.glob(f"{esc_dir}/data/*.parquet"))]
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["fold"].isin(allowed_folds)) & df["category"].isin(ESC_TO_VAANI)]
    return [
        {"source": "esc50", "recording_id": f"esc50:{row.filename}",
         "category": ESC_TO_VAANI[row.category], "audio": row.audio["bytes"],
         "source_label": row.category, "fold": int(row.fold)}
        for row in df.itertuples(index=False)
    ]


def vaani_events(records: list[dict]) -> list[dict]:
    out = []
    for rec in records:
        for index, event in enumerate(rec.get("events", [])):
            duration = float(event["offset"]) - float(event["onset"])
            category = event.get("category")
            if category in PRIORS and 0.05 <= duration <= 5.0:
                out.append({
                    "source": "vaani_gold", "recording_id": f"vaani:{rec['clip_id']}",
                    "category": category, "wav": rec["wav"], "event_index": index,
                    "onset": float(event["onset"]), "offset": float(event["offset"]),
                    "source_label": event.get("tag", ""), "fold": None,
                })
    return out


def load_official_bank(path: str) -> list[dict]:
    """The organisers' own noise recordings, recovered from the validation synthetic clips (noisy minus clean
    reference) and cut at the verified timestamps; each entry is one whole event at its natural duration."""
    out = []
    for i, b in enumerate(json.load(open(path))):
        out.append({"source": "official", "recording_id": f"official:{b['wav']}", "category": b["category"], "wav": b["wav"],
                    "onset": 0.0, "offset": float(b["duration"]), "source_label": b.get("tag") or "", "fold": None, "natural_duration": True})
    return out


def select_events(
    sources: list[dict], n_events: int, max_uses: int, rng: np.random.Generator,
) -> tuple[list[dict], dict]:
    by_category: dict[str, list[dict]] = defaultdict(list)
    for source in sources:
        by_category[source["category"]].append(source)
    missing = sorted(set(PRIORS) - set(by_category))
    available = [category for category in PRIORS if category in by_category]
    weights = np.array([PRIORS[category] for category in available], dtype=np.float64)
    weights /= weights.sum()
    order = {category: rng.permutation(len(by_category[category])).tolist() for category in available}
    cursor = Counter()
    uses = Counter()
    selected = []
    for _ in range(n_events):
        category = str(rng.choice(available, p=weights))
        candidates = by_category[category]
        found = None
        for _ in range(len(candidates)):
            source = candidates[order[category][cursor[category] % len(candidates)]]
            cursor[category] += 1
            if uses[source["recording_id"]] < max_uses:
                found = source
                break
        if found is None:
            raise RuntimeError(
                f"event bank exhausted for {category}; increase bank size or --max-uses"
            )
        uses[found["recording_id"]] += 1
        selected.append(found)
    return selected, {"missing_categories": missing, "available_categories": available}


def source_audio(source: dict, audio_root: str) -> np.ndarray:
    if source["source"] == "esc50":
        x, sr = sf.read(io.BytesIO(source["audio"]), dtype="float32")
    else:
        x, sr = sf.read(Path(audio_root) / source["wav"], dtype="float32")
        lo, hi = int(source["onset"] * sr), int(source["offset"] * sr)
        x = x[lo:hi]
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = resample(x, sr)
    if source["source"] == "esc50":
        lo, hi = energetic_span(x)
        x = x[lo:hi]
    return x


CLIP_PREFIX = "synthv2"


def build_one(job: tuple) -> dict | None:
    index, bed_rec, sources, audio_root, out_dir, seed, snr_lo, snr_hi, level_ref, peak_norm, bed_clean, over_sub, bed_p50, enh_root, gate_db, gate_atten, min_gap, cat_snr, cat_sd = job
    rng = np.random.default_rng(seed)
    x, sr = sf.read(Path(audio_root) / bed_rec["wav"], dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = resample(x, sr)
    if enh_root:
        # speech estimate from the Track 2 U-Net (mixture minus its residual): keeps speech frames at
        # natural level and removes the floor, which is the profile of the eval synthetic beds
        res_path = Path(enh_root) / bed_rec["wav"]
        if not res_path.exists():
            return None
        res, rsr = sf.read(res_path, dtype="float32")
        if res.ndim > 1:
            res = res.mean(axis=1)
        res = resample(res, rsr)
        n = min(len(x), len(res))
        x = x[:n] - res[:n]
    regions = free_regions(bed_rec)
    pieces = [x[int(lo * SR):int(hi * SR)] for lo, hi in regions]
    if not pieces:
        return None
    bed = np.concatenate(pieces)
    if len(bed) < 2 * SR:
        return None
    n_samples = min(len(bed), int(rng.uniform(4.0, 11.0) * SR))
    start = int(rng.integers(0, len(bed) - n_samples + 1))
    bed = bed[start:start + n_samples]
    if bed_clean:
        bed = make_synthetic(bed, rng, over_subtract=over_sub)
    if gate_db > 0:
        # noise gate: 50 ms frames more than gate_db below the bed's median level are attenuated by
        # gate_atten dB with linear ramps, reproducing the eval synthetic floor (p5 about 41 dB under p50)
        n_fr = len(bed) // 800
        if n_fr >= 4:
            fdb = 10 * np.log10((bed[:n_fr * 800].reshape(n_fr, 800) ** 2).mean(1) + 1e-12)
            g = np.where(fdb < np.median(fdb) - gate_db, 10 ** (-gate_atten / 20.0), 1.0)
            centres = (np.arange(n_fr) + 0.5) * 800
            bed = bed * np.interp(np.arange(len(bed)), centres, g).astype(np.float32)
    if bed_p50 is not None:
        # pin the bed's median 50 ms frame level (eval synthetic speech sits at natural level before the final peak normalisation)
        n_fr = len(bed) // 800
        fdb = 10 * np.log10((bed[:n_fr * 800].reshape(n_fr, 800) ** 2).mean(1) + 1e-12)
        bed = bed * 10 ** ((bed_p50 + rng.normal(0.0, 1.5) - float(np.median(fdb))) / 20.0)
    out, bed_rms, events = bed.copy(), rms(bed), []
    bed_peak = float(np.abs(bed).max()) + 1e-8
    for source in sources:
        if source["recording_id"] == f"vaani:{bed_rec['clip_id']}":
            return None
        event = source_audio(source, audio_root)
        duration = float(np.clip(rng.lognormal(mean=np.log(0.72), sigma=0.55), 0.2, 3.0))
        if source.get("natural_duration"):
            duration = len(event) / SR
        target = int(duration * SR)
        if len(event) < int(0.05 * SR):
            return None
        if len(event) > target:
            offset = int(rng.integers(0, len(event) - target + 1))
            event = event[offset:offset + target]
        elif len(event) < target:
            event = resize(event, target)
        if len(event) > len(out):
            event = event[:len(out)]
        pos = int(rng.integers(0, len(out) - len(event) + 1))
        if min_gap >= 0:
            # eval synthetic events are separable (3.8 detected per clip at F1 0.93), so placements must
            # not overlap; rejection-sample a position clear of every earlier event by min_gap seconds
            gap = int(min_gap * SR)
            placed = False
            for _ in range(60):
                pos = int(rng.integers(0, len(out) - len(event) + 1))
                if all(pos + len(event) + gap <= int(e["onset"] * SR) or pos >= int(e["offset"] * SR) + gap for e in events):
                    placed = True
                    break
            if not placed:
                continue
        if level_ref == "cat":
            # organisers' law (probe_off_synstruct): event RMS sits at a per-category offset from the clip's
            # speech RMS (human_non_speech -3.3 dB, every other category +6.4 dB) with a ~3 dB per-event spread
            mu = cat_snr.get(source["category"], cat_snr.get("*", 6.4))
            snr_db = float(mu + rng.normal(0.0, cat_sd))
        else:
            snr_db = float(rng.uniform(snr_lo, snr_hi))
        if level_ref == "abs":
            # bed keeps the natural clip's gain; the event peak sits r dB above full scale, so the
            # final peak normalisation attenuates the speech by r (eval synthetic: about 7 dB)
            event = event * (10 ** (snr_db / 20.0) / (float(np.abs(event).max()) + 1e-8))
        elif level_ref == "peak":
            # eval synthetic clips are peak-normalised with the bed only ~5 dB below full scale, so
            # events are placed by peak relative to the bed's peak rather than by RMS
            event = event * (bed_peak / (float(np.abs(event).max()) + 1e-8) * 10 ** (snr_db / 20.0))
        else:
            event = event * (bed_rms / rms(event) * 10 ** (snr_db / 20.0))
        out[pos:pos + len(event)] += event
        events.append({
            "onset": round(pos / SR, 4), "offset": round((pos + len(event)) / SR, 4),
            "category": source["category"], "tag": f"<{source['category']}>",
            "provenance": {key: source[key] for key in (
                "source", "recording_id", "source_label", "fold", "wav", "event_index",
                "onset", "offset",
            ) if key in source},
            "snr_db": round(snr_db, 4),
        })
    peak = float(np.abs(out).max())
    if peak_norm > 0:
        out *= peak_norm / max(peak, 1e-8)
    elif peak > 0.999:
        out *= 0.999 / peak
    clip_id = f"{CLIP_PREFIX}_{index:06d}"
    shard = f"{index // 2000:03d}"
    wav = Path(out_dir) / "wav" / shard / f"{clip_id}.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(wav, out, SR, subtype="PCM_16")
    events.sort(key=lambda event: event["onset"])
    return {
        "clip_id": clip_id, "wav": f"wav/{shard}/{clip_id}.wav",
        "duration": round(len(out) / SR, 4), "sr": SR, "tier": "synth",
        "state": bed_rec.get("state", ""), "district": bed_rec.get("district", ""),
        "language": bed_rec.get("language", ""), "speaker": bed_rec["speaker"],
        "categories": sorted({event["category"] for event in events}), "transcript": "",
        "events": events, "synthetic": True,
        "bed_provenance": {"recording_id": f"vaani:{bed_rec['clip_id']}", "wav": bed_rec["wav"]},
    }


def main() -> None:
    global CLIP_PREFIX
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=("train", "val"), required=True)
    ap.add_argument("--beds", required=True, help="Split manifest used exclusively for beds and Vaani events")
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--esc-dir", default=f"{DATA}/esc50")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--max-uses", type=int, default=4)
    ap.add_argument("--attempt-multiplier", type=float, default=1.0,
                    help="Generate extra candidates to compensate for short free bed regions")
    ap.add_argument("--snr-lo", type=float, default=0.0)
    ap.add_argument("--snr-hi", type=float, default=10.0)
    ap.add_argument("--level-ref", choices=("rms", "peak", "abs", "cat"), default="rms",
                    help="scale events relative to the bed's RMS (v2) or its peak (matches the eval synthetic level)")
    ap.add_argument("--cat-snr", default="human_non_speech:-3.3,*:6.4",
                    help="per-category event RMS offset from the bed RMS in dB for --level-ref cat (* = default)")
    ap.add_argument("--cat-snr-sd", type=float, default=3.0, help="per-event spread in dB for --level-ref cat")
    ap.add_argument("--prior-boost", default=None, help="category prior multipliers, e.g. phone_signal_alarm:3 (renormalised)")
    ap.add_argument("--short-boost", type=float, default=1.0, help="sampling multiplier for bank events shorter than --short-sec")
    ap.add_argument("--short-sec", type=float, default=0.25)
    ap.add_argument("--peak-norm", type=float, default=0.0, help="normalise every clip to this peak (eval synthetic uses 0.99); 0 keeps v2 behaviour")
    ap.add_argument("--clip-prefix", default="synthv2")
    ap.add_argument("--official-bank", default=None, help="bank.json recovered from the organisers' validation synthetic clips; replaces the ESC/Vaani banks and their prior")
    ap.add_argument("--count-probs", default=None, help="probabilities for 2,3,4,5,6 events per clip (comma-separated)")
    ap.add_argument("--over-subtract", type=float, default=2.5, help="spectral over-subtraction factor for the clean bed")
    ap.add_argument("--bed-p50", type=float, default=None, help="pin the bed's median frame level (dB) before events are added; eval-matched value -18")
    ap.add_argument("--bed-enhanced-root", default=None, help="residual root; bed = mixture minus residual (U-Net speech estimate), implies --no-clean-bed")
    ap.add_argument("--bed-gate-db", type=float, default=0.0, help="noise-gate threshold below the bed's median frame level (0 = off)")
    ap.add_argument("--bed-gate-atten", type=float, default=26.0, help="noise-gate attenuation in dB")
    ap.add_argument("--min-gap", type=float, default=-1.0, help="minimum gap between events in seconds; negative keeps v2's independent (overlapping) placement")
    ap.add_argument("--no-clean-bed", action="store_true",
                    help="keep the bed at its natural level and floor (eval synthetic beds are attenuated natural audio, not denoised)")
    args = ap.parse_args()
    CLIP_PREFIX = args.clip_prefix
    cat_snr = {k: float(v) for k, v in (kv.split(":") for kv in args.cat_snr.split(","))}
    if args.max_uses < 1 or args.n < 1 or args.attempt_multiplier < 1 or args.snr_hi < args.snr_lo:
        raise ValueError("invalid --max-uses, --n, --attempt-multiplier, or SNR range")
    records = [json.loads(line) for line in open(args.beds) if line.strip()]
    gold = [record for record in records if record["tier"] == "gold" and record.get("events")]
    beds = [
        record for record in gold
        if sum(hi - lo for lo, hi in free_regions(record)) >= 2.0
    ]
    esc_folds = {1, 2, 3, 4} if args.split == "train" else {5}
    if args.official_bank:
        bank = load_official_bank(args.official_bank)
        # category prior = the organisers' own bank frequencies
        counts_c = Counter(b["category"] for b in bank); tot = sum(counts_c.values())
        for c in list(PRIORS): PRIORS[c] = counts_c.get(c, 0) / tot
        if args.prior_boost:
            for kv in args.prior_boost.split(","):
                k, v = kv.split(":"); PRIORS[k] = PRIORS.get(k, 0.0) * float(v)
            tot = sum(PRIORS.values())
            for c in list(PRIORS): PRIORS[c] /= tot
        if args.short_boost > 1.0:
            # the pair's misses concentrate on sub-0.25 s events (offval_B2): sample them more often
            short = [b for b in bank if b["offset"] < args.short_sec]
            bank = bank + short * int(round(args.short_boost - 1.0))
    else:
        bank = vaani_events(gold) + load_esc(args.esc_dir, esc_folds)
    rng = np.random.default_rng(args.seed)
    n_candidates = int(np.ceil(args.n * args.attempt_multiplier))
    count_p = [float(x) for x in args.count_probs.split(",")] if args.count_probs else [0.10, 0.25, 0.30, 0.25, 0.10]
    counts = rng.choice(np.arange(2, 7), size=n_candidates, p=count_p)
    selected, availability = select_events(bank, int(counts.sum()), args.max_uses, rng)
    offsets = np.cumsum(np.r_[0, counts])
    selected_by_clip = [selected[offsets[i]:offsets[i + 1]] for i in range(n_candidates)]
    chosen_beds = []
    for clip_sources in selected_by_clip:
        source_ids = {source["recording_id"] for source in clip_sources}
        for _ in range(100):
            bed = beds[int(rng.integers(len(beds)))]
            if f"vaani:{bed['clip_id']}" not in source_ids:
                chosen_beds.append(bed)
                break
        else:
            raise RuntimeError("could not select a bed distinct from its Vaani event sources")
    jobs = [
        (i, chosen_beds[i], selected_by_clip[i], args.audio_root, args.out,
         args.seed + i, args.snr_lo, args.snr_hi, args.level_ref, args.peak_norm, not args.no_clean_bed, args.over_subtract, args.bed_p50, args.bed_enhanced_root, args.bed_gate_db, args.bed_gate_atten, args.min_gap, cat_snr, args.cat_snr_sd)
        for i in range(n_candidates)
    ]
    Path(args.out).mkdir(parents=True, exist_ok=True)
    written = 0
    with open(Path(args.out) / "manifest.jsonl", "w") as manifest:
        with mp.Pool(args.workers) as pool:
            for result in pool.imap_unordered(build_one, jobs, chunksize=8):
                if result is not None:
                    manifest.write(json.dumps(result) + "\n")
                    written += 1
                    if written == args.n:
                        break
    meta = {
        "split": args.split, "n_requested": args.n, "n_candidates": n_candidates,
        "n_written": written,
        "esc50_folds": sorted(esc_folds), "max_uses": args.max_uses,
        "snr_range_db": [args.snr_lo, args.snr_hi], "bank_availability": availability,
        "bank_recordings": dict(Counter(source["source"] for source in bank)),
        "source_uses": dict(Counter(source["recording_id"] for source in selected)),
    }
    with open(Path(args.out) / "build_metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    print(f"wrote {written}/{args.n} clips to {args.out}")


if __name__ == "__main__":
    main()
