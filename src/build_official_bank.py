"""Recover the organisers' noise recordings from the validation synthetic clips (noisy minus clean reference), cut them at the
verified event timestamps into a categorised bank, and report the generator's own statistics (event level vs speech, durations, gaps)."""
import json, os, collections, numpy as np, soundfile as sf

from paths import DATA

V = f"{DATA}/official_val/validation"
OUT = f"{DATA}/official_noise_bank"
os.makedirs(OUT, exist_ok=True)
recs = [json.loads(l) for l in open(f"{DATA}/splits/official_val_syn.jsonl")]
bank, stats = [], collections.defaultdict(list)
resid_out, resid_in = [], []
for r in recs:
    fn = os.path.basename(r["wav"])
    x, sr = sf.read(f"{V}/syntheticNoiseAudio/{fn}", dtype="float32"); c, sr2 = sf.read(f"{V}/syntheticCleanRefAudio/{fn}", dtype="float32")
    x = x.mean(1) if x.ndim > 1 else x; c = c.mean(1) if c.ndim > 1 else c
    n = min(len(x), len(c)); x, c = x[:n], c[:n]
    # the noisy clip may be rescaled (peak-normalised) relative to the clean reference: fit one gain on the event-free frames
    mask = np.ones(n, dtype=bool)
    for e in r["events"]:
        mask[int(e["onset"] * sr): int(e["offset"] * sr)] = False
    g = float(np.dot(x[mask], c[mask]) / (np.dot(c[mask], c[mask]) + 1e-9)) if mask.sum() > 1600 else 1.0
    noise = x - g * c
    fr = lambda a: 10 * np.log10(np.mean(a ** 2) + 1e-12)
    resid_out.append(fr(noise[mask])); resid_in.append(fr(noise[~mask]) if (~mask).sum() else np.nan)
    sp_peak = 20 * np.log10(np.abs(g * c).max() + 1e-9); sp_rms = fr(g * c)
    evs = sorted(r["events"], key=lambda e: e["onset"])
    for i, e in enumerate(evs):
        a, b = int(e["onset"] * sr), int(e["offset"] * sr)
        if b - a < 800: continue
        seg = noise[a:b]
        d = f"{OUT}/{e['category']}"; os.makedirs(d, exist_ok=True)
        p = f"{d}/{os.path.splitext(fn)[0]}_{i}.wav"; sf.write(p, seg, sr, subtype="PCM_16")
        bank.append({"wav": p, "category": e["category"], "tag": e.get("tag"), "duration": round((b - a) / sr, 4), "clip": fn, "gain_fit": round(g, 4)})
        stats["dur"].append((b - a) / sr)
        stats["peak_rel_speech_peak_db"].append(20 * np.log10(np.abs(seg).max() + 1e-9) - sp_peak)
        stats["rms_rel_speech_rms_db"].append(fr(seg) - sp_rms)
        if i: stats["gap"].append(e["onset"] - evs[i - 1]["offset"])
    stats["n_events"].append(len(evs)); stats["clip_dur"].append(r["duration"]); stats["gain_fit"].append(g)
json.dump(bank, open(f"{OUT}/bank.json", "w"), indent=1)
print(f"bank: {len(bank)} noise events from {len(recs)} clips; categories {collections.Counter(b['category'] for b in bank)}")
q = lambda a: np.percentile(a, [10, 50, 90])
print(f"residual energy outside events (subtraction quality): median {np.median(resid_out):.1f} dB; inside events {np.nanmedian(resid_in):.1f} dB; gain fit median {np.median(stats['gain_fit']):.3f}")
print(f"events per clip {np.mean(stats['n_events']):.2f}; clip duration p10/50/90 {q(stats['clip_dur'])}; event duration {q(stats['dur'])}; gap {q(stats['gap'])}")
print(f"event peak rel. speech peak dB p10/50/90 {q(stats['peak_rel_speech_peak_db'])}; event RMS rel. speech RMS dB {q(stats['rms_rel_speech_rms_db'])}")
ov = sum(1 for g in stats["gap"] if g < 0); print(f"overlapping consecutive events: {ov} of {len(stats['gap'])}")
