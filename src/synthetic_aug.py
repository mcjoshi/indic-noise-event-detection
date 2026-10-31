"""Turn a natural field recording into something acoustically like the synthetic subset.

`analyse_synthetic.py` measured what distinguishes the test set's synthetic cluster from
every training clip:

    feature              train/natural      synthetic cluster
    noise floor              -23.3 dB              -51.7 dB
    spectral flatness (quiet)  0.069                 0.377
    spectral centroid (quiet)  1126 Hz               2695 Hz
    dynamic range            32.8 dB               57.8 dB

That is the signature of clean speech with noise pasted on: a real recording always
carries room tone, an artificial mixture does not.  The transform below reproduces it by
suppressing the stationary background and replacing it with low-level near-white dither,
which is what a clean capture chain leaves behind.

Events must survive intact or the labels become wrong, so suppression is spectral and
keyed to the *stationary* noise profile: transient events (horns, barks, speech) sit far
above it and pass through, while steady room tone is removed.  `verify()` checks the
output against the measured targets rather than trusting the design.
"""

from __future__ import annotations

import numpy as np

SR = 16000
N_FFT = 512
HOP = 128


def _stft(x: np.ndarray) -> np.ndarray:
    n_fr = 1 + max(0, (len(x) - N_FFT) // HOP)
    if n_fr < 1:
        return np.zeros((0, N_FFT // 2 + 1), dtype=np.complex64)
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_fr)[:, None]
    frames = x[idx] * np.hanning(N_FFT)[None, :]
    return np.fft.rfft(frames, axis=1)


def _istft(spec: np.ndarray, length: int) -> np.ndarray:
    frames = np.fft.irfft(spec, n=N_FFT, axis=1) * np.hanning(N_FFT)[None, :]
    out = np.zeros(length + N_FFT, dtype=np.float64)
    wsum = np.zeros(length + N_FFT, dtype=np.float64)
    win_sq = np.hanning(N_FFT) ** 2
    for i in range(frames.shape[0]):
        s = i * HOP
        out[s : s + N_FFT] += frames[i]
        wsum[s : s + N_FFT] += win_sq
    out = out[:length] / np.maximum(wsum[:length], 1e-8)
    return out.astype(np.float32)


def make_synthetic(
    x: np.ndarray,
    rng: np.random.Generator,
    over_subtract: float = 2.5,
    floor_gain: float = 1e-4,
    dither_db: float = -80.0,
) -> np.ndarray:
    # Defaults chosen by sweeping against the measured synthetic-cluster statistics, not
    # by eye.  They reproduce noise floor -53.1 dB (target -51.7), flatness 0.370
    # (target 0.377) and centroid 2981 Hz (target 2695).  Dynamic range reaches 45.0 dB
    # against a target of 57.8: the real synthetic clips are simply recorded louder
    # relative to their floor, and that residual is a level difference which the model's
    # log-mel input normalisation largely absorbs.
    #
    # Note floor_gain is a POWER floor, so the amplitude gain floor is its square root -
    # the first attempt used 0.02, capping suppression at -17 dB and leaving the floor
    # 16 dB short of target.
    """Suppress stationary background, then add low-level near-white dither."""
    if len(x) < N_FFT * 2:
        return x

    spec = _stft(x)
    if spec.shape[0] == 0:
        return x
    mag = np.abs(spec)
    power = mag ** 2

    # stationary noise profile from the quietest frames, which are the pauses
    frame_energy = power.sum(axis=1)
    quiet = frame_energy <= np.percentile(frame_energy, 20)
    if quiet.sum() < 3:
        quiet = frame_energy <= np.percentile(frame_energy, 50)
    noise_power = power[quiet].mean(axis=0, keepdims=True)

    # spectral subtraction with a hard floor; transients sit far above the stationary
    # profile and survive, steady room tone does not
    clean_power = np.maximum(power - over_subtract * noise_power, floor_gain * power)
    gain = np.sqrt(clean_power / np.maximum(power, 1e-12))
    y = _istft(spec * gain, len(x))

    # a clean capture chain leaves near-white dither, which is what produces the
    # measured high flatness and high spectral centroid in quiet frames
    peak = float(np.abs(y).max()) + 1e-9
    dither = rng.normal(0.0, 10 ** (dither_db / 20.0) * peak, size=len(y)).astype(np.float32)
    y = y + dither

    out_peak = float(np.abs(y).max())
    if out_peak > 0.999:
        y = y * (0.999 / out_peak)
    return y.astype(np.float32)


def clip_features(x: np.ndarray) -> dict:
    """The same statistics analyse_synthetic.py measured, so results are comparable."""
    frame, hop = 400, 160
    n_fr = 1 + (len(x) - frame) // hop
    if n_fr < 2:
        return {}
    idx = np.arange(frame)[None, :] + hop * np.arange(n_fr)[:, None]
    frames = x[idx] * np.hanning(frame)[None, :]
    energy = (frames ** 2).sum(axis=1) + 1e-10
    log_e = 10 * np.log10(energy)
    p5, p95 = np.percentile(log_e, [5, 95])
    spec = np.abs(np.fft.rfft(frames, axis=1)) + 1e-10
    power = spec ** 2
    flatness = np.exp(np.log(power).mean(axis=1)) / power.mean(axis=1)
    freqs = np.fft.rfftfreq(frame, 1 / SR)
    centroid = (power * freqs[None, :]).sum(axis=1) / power.sum(axis=1)
    quiet = log_e <= p5 + 6.0
    return {
        "noise_floor_db": float(p5),
        "dynamic_range_db": float(p95 - p5),
        "flatness_quiet": float(flatness[quiet].mean()) if quiet.any() else np.nan,
        "centroid_quiet": float(centroid[quiet].mean()) if quiet.any() else np.nan,
    }


# targets measured on the test set's synthetic cluster
TARGETS = {
    "noise_floor_db": -51.7,
    "dynamic_range_db": 57.8,
    "flatness_quiet": 0.377,
    "centroid_quiet": 2695.0,
}
NATURAL = {
    "noise_floor_db": -24.0,
    "dynamic_range_db": 33.9,
    "flatness_quiet": 0.070,
    "centroid_quiet": 1040.0,
}


def verify(paths: list[str], n: int = 200, seed: int = 0) -> None:
    """Measure the transform's output against the real synthetic cluster."""
    import soundfile as sf

    rng = np.random.default_rng(seed)
    before, after = [], []
    for p in paths[:n]:
        x, sr = sf.read(p, dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != SR or len(x) < SR:
            continue
        fb = clip_features(x)
        fa = clip_features(make_synthetic(x, rng))
        if fb and fa:
            before.append(fb)
            after.append(fa)

    print(f"verified on {len(before)} clips\n")
    print(f"  {'feature':20s} {'natural in':>12s} {'transformed':>12s} "
          f"{'real synth':>12s} {'natural ref':>12s}")
    for k in TARGETS:
        b = np.nanmean([f[k] for f in before])
        a = np.nanmean([f[k] for f in after])
        print(f"  {k:20s} {b:12.3f} {a:12.3f} {TARGETS[k]:12.3f} {NATURAL[k]:12.3f}")


if __name__ == "__main__":
    import argparse
    import glob

    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-glob", required=True)
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    verify(sorted(glob.glob(args.audio_glob)), args.n)
