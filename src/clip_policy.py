"""Per-clip activity threshold for the threshold routes: a gradient-boosted regressor on eight clip-level features of the
smoothed activity curve predicts the clip's Dice-best threshold (the report's cross-fitted policy, +0.0044 Dice on both official folds)."""
import numpy as np
from model import median_filter
from scorer import frame_mask

def features(probs: np.ndarray, median_k: int, duration: float) -> list[float]:
    a = median_filter(probs.astype(np.float64), median_k)
    return [duration, float(a.mean()), float(a.max()), float(np.median(a)), float((a > 0.5).mean()), float(np.percentile(a, 90)), float(a.std()),
            float(((a[1:] >= 0.5) & (a[:-1] < 0.5)).sum())]

def apply(policy: dict, probs: np.ndarray, duration: float) -> float:
    x = np.array([features(probs, policy["median_k"], duration)])
    return float(np.clip(policy["model"].predict(x)[0], policy["tmin"], policy["tmax"]))

def best_threshold(probs, refs, duration, median_k, grid):
    n = probs.shape[0]; gm = frame_mask(refs, n) if refs else np.zeros(n, dtype=bool)
    a = median_filter(probs.astype(np.float64), median_k); best, bt = -1.0, grid[len(grid) // 2]
    for t in grid:
        pm = a >= t; inter = np.logical_and(gm, pm).sum(); denom = gm.sum() + pm.sum()
        d = 1.0 if denom == 0 else 2.0 * inter / denom
        if d > best: best, bt = d, t
    return float(bt)

def fit(probs: dict, records: list, durations: dict, median_k: int, grid, folds=None) -> dict:
    from sklearn.ensemble import GradientBoostingRegressor
    refs = {r["clip_id"]: [(e["onset"], e["offset"]) for e in r["events"]] for r in records}
    ids = [c for c in refs if c in probs]
    X = {c: features(probs[c], median_k, durations[c]) for c in ids}; y = {c: best_threshold(probs[c], refs[c], durations[c], median_k, grid) for c in ids}
    m = GradientBoostingRegressor(random_state=0).fit(np.array([X[c] for c in ids]), np.array([y[c] for c in ids]))
    return {"model": m, "median_k": median_k, "tmin": float(min(grid)), "tmax": float(max(grid))}
