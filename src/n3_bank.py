"""Build the N3 insert bank: each ESC-50 recording trimmed to its longest active segment at 16 kHz."""
from __future__ import annotations

import glob
import io

import numpy as np
import pandas as pd
import scipy.signal as ss
import soundfile as sf

from paths import DATA

SR = 16000


def active_segment(x: np.ndarray, hop: int = 160, merge_gap: int = 10) -> tuple[int, int] | None:
    n = len(x) // hop
    if n < 5:
        return None
    rms = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    on = db > max(db.max() - 25.0, np.percentile(db, 10) + 6.0)
    idx = np.flatnonzero(on)
    if idx.size == 0:
        return None
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > merge_gap:
            runs.append((start, prev + 1)); start = i
        prev = i
    runs.append((start, prev + 1))
    a, b = max(runs, key=lambda r: r[1] - r[0])
    return (a * hop, b * hop) if b - a >= 10 else None


def main() -> None:
    df = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{DATA}/esc50/data/*.parquet"))])
    chunks, offs, cats, lens = [], [0], [], []
    for _, row in df.iterrows():
        x, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        x = x.mean(1) if x.ndim > 1 else x
        x = ss.resample_poly(x, SR, sr).astype(np.float32) if sr != SR else x
        seg = active_segment(x)
        if seg is None:
            continue
        y = x[seg[0]: seg[1]]
        y = y / (np.sqrt(np.mean(y ** 2)) + 1e-8)
        chunks.append(y.astype(np.float16)); offs.append(offs[-1] + len(y)); cats.append(row["category"]); lens.append(len(y) / SR)
    np.savez(DATA / "n3_esc_bank.npz", audio=np.concatenate(chunks), offsets=np.array(offs), category=np.array(cats))
    print(f"{len(chunks)} inserts, duration median {np.median(lens):.2f} s, p10 {np.percentile(lens, 10):.2f}, p90 {np.percentile(lens, 90):.2f}")


if __name__ == "__main__":
    main()
