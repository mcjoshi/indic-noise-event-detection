"""Turn the organisers' validation release into usable Track 1 and Track 2 validation sets.

Track 1: 2,501 clips with verified timestamps in the evaluation distribution - the first
in-distribution labelled data we have had. Track 2: 695 (mix, clean) pairs built from the
organisers' own noise, replacing our ESC-50 mixtures that carried a ~7 dB home-field bias.
"""
import json
import soundfile as sf

from paths import DATA

V = DATA / "official_val/validation"
OUT = DATA / "official_val/prepared"
SR = 16000
meta = json.load(open(V / "validationMetadata.json"))

(OUT / "t2_mix").mkdir(parents=True, exist_ok=True)
(OUT / "t2_clean").mkdir(parents=True, exist_ok=True)
(OUT / "t1_audio").mkdir(parents=True, exist_ok=True)

t1, t2, miss = [], [], 0
for m in meta:
    cid = m["segmentFileName"][:-4]
    syn = bool(m.get("syntheticData"))
    if syn:
        c, n = V / "syntheticCleanRefAudio" / f"{cid}.wav", V / "syntheticNoiseAudio" / f"{cid}.wav"
        if not (c.exists() and n.exists()):
            miss += 1; continue
        a, _ = sf.read(c, dtype="float32"); b, _ = sf.read(n, dtype="float32")
        if a.ndim > 1: a = a.mean(axis=1)
        if b.ndim > 1: b = b.mean(axis=1)
        k = min(len(a), len(b)); mix = a[:k] + b[:k]
        sf.write(OUT / "t2_mix" / f"{cid}.wav", mix, SR, subtype="PCM_16")
        sf.write(OUT / "t2_clean" / f"{cid}.wav", a[:k], SR, subtype="PCM_16")
        sf.write(OUT / "t1_audio" / f"{cid}.wav", mix, SR, subtype="PCM_16")
        t2.append({"clip_id": cid, "mix_wav": f"t2_mix/{cid}.wav",
                   "clean_wav": f"t2_clean/{cid}.wav"})
    else:
        p = V / "naturalNoisyAudio" / f"{cid}.wav"
        if not p.exists():
            miss += 1; continue
        x, _ = sf.read(p, dtype="float32")
        if x.ndim > 1: x = x.mean(axis=1)
        sf.write(OUT / "t1_audio" / f"{cid}.wav", x, SR, subtype="PCM_16")
    evs = [{"onset": float(e["start"]), "offset": float(e["end"])}
           for e in (m.get("NoiseSubCategoryTimeStamp") or [])]
    t1.append({"clip_id": cid, "wav": f"t1_audio/{cid}.wav", "synthetic": syn,
               "duration": m.get("duration"), "events": evs})

with open(OUT / "t1_manifest.jsonl", "w") as f:
    for r in t1: f.write(json.dumps(r) + "\n")
with open(OUT / "t2_manifest.jsonl", "w") as f:
    for r in t2: f.write(json.dumps(r) + "\n")
ns = sum(1 for r in t1 if r["synthetic"])
print(f"Track 1 val: {len(t1)} clips ({ns} synthetic, {len(t1)-ns} natural), "
      f"{sum(len(r['events']) for r in t1)} verified events")
print(f"Track 2 val: {len(t2)} (mix, clean) pairs from the organisers' own noise")
print(f"missing files skipped: {miss}")
print(f"-> {OUT}")
