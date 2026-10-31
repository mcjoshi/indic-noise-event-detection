"""Is the synthetic test subset actually a distinct acoustic population?

The test set is stated to be 62% natural / 38% synthetic (clean speech + programmatically
added noise), but participants are not given the `syntheticData` flag, and no training
clip is synthetic.  Before building any synthetic-specific augmentation it is worth
establishing empirically whether:

  1. the test set separates into two populations at roughly the stated 62/38 ratio, and
  2. the training distribution covers only one of them.

If both hold, synthetic-condition training is justified and we additionally get a
detector that can route decoding per clip.  If the split is not detectable, the two
conditions are acoustically similar and separate treatment buys nothing.

Features are chosen to expose artificial mixing rather than content:
  - noise floor and dynamic range: an artificial mix of clean speech with added noise
    tends to have a quieter, flatter floor between events than a phone recorded outdoors
  - reverberation decay: real recordings couple speech and background through one room
    impulse response; an artificial mix does not
  - spectral flatness / rolloff in low-energy frames: captures the character of the
    background itself
  - digital silence and clipping: mixing pipelines often leave exact zeros or clip
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import warnings

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")

SR = 16000
FRAME = 400
HOP = 160
EPS = 1e-10


def features(path: str) -> dict | None:
    try:
        x, sr = sf.read(path, dtype="float32")
    except Exception:
        return None
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR or len(x) < SR // 2:
        return None

    n_fr = 1 + (len(x) - FRAME) // HOP
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n_fr)[:, None]
    frames = x[idx] * np.hanning(FRAME)[None, :]

    energy = (frames ** 2).sum(axis=1) + EPS
    log_e = 10 * np.log10(energy)
    p5, p50, p95 = np.percentile(log_e, [5, 50, 95])

    spec = np.abs(np.fft.rfft(frames, axis=1)) + EPS
    power = spec ** 2
    # spectral flatness: geometric over arithmetic mean
    flatness = np.exp(np.log(power).mean(axis=1)) / power.mean(axis=1)
    freqs = np.fft.rfftfreq(FRAME, 1 / SR)
    centroid = (power * freqs[None, :]).sum(axis=1) / power.sum(axis=1)
    cum = np.cumsum(power, axis=1)
    rolloff = freqs[np.argmax(cum >= 0.95 * cum[:, -1:], axis=1)]

    quiet = log_e <= p5 + 6.0     # the quietest frames: the background itself
    loud = log_e >= p95 - 6.0

    # reverberation proxy: how fast energy decays after loud frames.  A real room gives
    # a gradual tail; an artificial mix of a clean source drops off abruptly.
    decays = []
    loud_idx = np.flatnonzero(loud)
    for i in loud_idx[:200]:
        if i + 6 < n_fr:
            drop = log_e[i] - log_e[i + 6]
            if np.isfinite(drop):
                decays.append(drop)

    return {
        "path": path,
        "clip_id": os.path.basename(path)[:-4],
        "duration": len(x) / SR,
        "noise_floor_db": float(p5),
        "median_db": float(p50),
        "dynamic_range_db": float(p95 - p5),
        "flatness_quiet": float(flatness[quiet].mean()) if quiet.any() else float("nan"),
        "flatness_all": float(flatness.mean()),
        "centroid_quiet": float(centroid[quiet].mean()) if quiet.any() else float("nan"),
        "rolloff_quiet": float(rolloff[quiet].mean()) if quiet.any() else float("nan"),
        "decay_db": float(np.mean(decays)) if decays else float("nan"),
        "frac_exact_zero": float((x == 0).mean()),
        "max_abs": float(np.abs(x).max()),
        "frac_clipped": float((np.abs(x) >= 0.999).mean()),
        "dc_offset": float(x.mean()),
    }


FEATURE_KEYS = [
    "noise_floor_db", "dynamic_range_db", "flatness_quiet", "flatness_all",
    "centroid_quiet", "rolloff_quiet", "decay_db", "frac_exact_zero", "max_abs",
]


def collect(paths: list[str], workers: int) -> list[dict]:
    with mp.Pool(workers) as pool:
        out = pool.map(features, paths, chunksize=32)
    return [r for r in out if r is not None]


def matrix(rows: list[dict]) -> np.ndarray:
    return np.array([[r[k] for k in FEATURE_KEYS] for r in rows], dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--train-manifest", required=True)
    ap.add_argument("--train-audio-root", required=True)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    test_paths = sorted(glob.glob(os.path.join(args.test_dir, "*.wav")))
    print(f"test clips: {len(test_paths)}")

    train_recs = [json.loads(l) for l in open(args.train_manifest) if l.strip()]
    rng = np.random.default_rng(0)
    sample = rng.choice(len(train_recs), size=min(args.n_train, len(train_recs)), replace=False)
    train_paths = [os.path.join(args.train_audio_root, train_recs[i]["wav"]) for i in sample]
    print(f"train sample: {len(train_paths)}")

    print("extracting features...", flush=True)
    test_rows = collect(test_paths, args.workers)
    train_rows = collect(train_paths, args.workers)
    print(f"  test {len(test_rows)}, train {len(train_rows)}")

    Xt, Xr = matrix(test_rows), matrix(train_rows)
    ok = np.isfinite(Xt).all(axis=1)
    Xt, test_rows = Xt[ok], [r for r, k in zip(test_rows, ok) if k]
    ok = np.isfinite(Xr).all(axis=1)
    Xr = Xr[ok]

    # standardise on the TRAIN statistics: train is known to be 100% natural, so it is
    # the reference distribution and any test-only mode shows up as a departure from it
    mu, sd = Xr.mean(axis=0), Xr.std(axis=0) + 1e-9
    Zt, Zr = (Xt - mu) / sd, (Xr - mu) / sd

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    km = KMeans(n_clusters=2, n_init=20, random_state=0).fit(Zt)
    lab = km.labels_
    frac = np.bincount(lab, minlength=2) / len(lab)

    # which test cluster do the (all-natural) training clips look like?
    d_train = np.linalg.norm(Zr[:, None, :] - km.cluster_centers_[None, :, :], axis=2)
    train_lab = d_train.argmin(axis=1)
    train_frac = np.bincount(train_lab, minlength=2) / len(train_lab)

    print("\n=== 2-cluster split of the TEST set ===")
    print(f"  cluster 0: {frac[0]:.1%}   cluster 1: {frac[1]:.1%}")
    print("  stated split: 62.3% natural / 37.7% synthetic")
    print("\n=== where the (100% natural) TRAIN sample lands ===")
    print(f"  cluster 0: {train_frac[0]:.1%}   cluster 1: {train_frac[1]:.1%}")

    print("\n=== per-feature means (train vs test clusters) ===")
    print(f"  {'feature':20s} {'train':>10s} {'test c0':>10s} {'test c1':>10s} {'c1-c0 (sd)':>12s}")
    for i, k in enumerate(FEATURE_KEYS):
        a, b = Xt[lab == 0, i].mean(), Xt[lab == 1, i].mean()
        print(f"  {k:20s} {Xr[:, i].mean():10.4f} {a:10.4f} {b:10.4f} {(b - a) / sd[i]:12.2f}")

    pca = PCA(n_components=2).fit(Zr)
    print("\n=== PCA (fitted on train) mean position ===")
    for name, Z in (("train", Zr), ("test c0", Zt[lab == 0]), ("test c1", Zt[lab == 1])):
        p = pca.transform(Z)
        print(f"  {name:10s} PC1 {p[:, 0].mean():7.3f}  PC2 {p[:, 1].mean():7.3f}")

    # Mahalanobis-style novelty: how far is each test clip from the train distribution?
    dist_t = np.linalg.norm(Zt, axis=1)
    dist_r = np.linalg.norm(Zr, axis=1)
    thresh = np.percentile(dist_r, 95)
    print("\n=== distance from the train distribution ===")
    print(f"  train  p50 {np.median(dist_r):.2f}  p95 {thresh:.2f}")
    print(f"  test   p50 {np.median(dist_t):.2f}  "
          f"fraction beyond train p95: {(dist_t > thresh).mean():.1%}")
    print(f"  test c0 beyond train p95: {(dist_t[lab == 0] > thresh).mean():.1%}")
    print(f"  test c1 beyond train p95: {(dist_t[lab == 1] > thresh).mean():.1%}")

    with open(args.out, "w") as fh:
        json.dump({
            "cluster_fraction": frac.tolist(),
            "train_cluster_fraction": train_frac.tolist(),
            "assignments": {r["clip_id"]: int(l) for r, l in zip(test_rows, lab)},
            "feature_keys": FEATURE_KEYS,
            "centers": km.cluster_centers_.tolist(),
            "train_mu": mu.tolist(), "train_sd": sd.tolist(),
        }, fh)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
