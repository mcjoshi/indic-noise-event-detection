"""Exploratory analysis from early development, not used by either released system. It reads unlabelled
test audio only.

Supervised natural/synthetic detector, calibrated to the published test prevalence.

Routing decode configs by condition only pays if the routing is accurate.  Unsupervised
k-means over the same features recovered 27.4% of test as synthetic against a stated
37.7%, so roughly 570 synthetic clips would take the natural route and lose the benefit.

Two improvements over k-means:
  1. Supervision - `val_mixed` has 3,239 clips whose condition is known by construction,
     so a logistic regression can be fitted rather than guessing at cluster geometry.
  2. Prevalence calibration - the organisers publish the split (2,080 synthetic of
     5,517).  The decision threshold is set so the flagged count matches that, which uses
     only stated public information and is strictly better than an arbitrary 0.5 cut.

The detector is evaluated on held-out val speakers before use, and routing must be scored
with *these* assignments rather than ground-truth flags: an oracle-routed score is an
upper bound, not an estimate.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from analyse_synthetic import FEATURE_KEYS, collect, features  # noqa: F401

TEST_SYNTHETIC_COUNT = 2080      # stated in dataset_info.txt
TEST_TOTAL = 5517


def matrix(rows: list[dict]) -> np.ndarray:
    return np.array([[r[k] for k in FEATURE_KEYS] for r in rows], dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-manifest", required=True)
    ap.add_argument("--val-audio-root", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()

    val = [json.loads(l) for l in open(args.val_manifest) if l.strip()]
    val_paths, val_y, val_spk, val_ids = [], [], [], []
    for r in val:
        p = r["wav"]
        val_paths.append(p if os.path.isabs(p) else os.path.join(args.val_audio_root, p))
        val_y.append(1 if r.get("synthetic") else 0)
        val_spk.append(r["speaker"])
        val_ids.append(r["clip_id"])
    print(f"val: {len(val_paths)} clips, {sum(val_y)} synthetic")

    print("extracting val features...", flush=True)
    val_rows = collect(val_paths, args.workers)
    keep = {r["path"]: r for r in val_rows}
    idx = [i for i, p in enumerate(val_paths) if p in keep]
    Xv = matrix([keep[val_paths[i]] for i in idx])
    yv = np.array([val_y[i] for i in idx])
    spk = np.array([val_spk[i] for i in idx])
    ids_v = [val_ids[i] for i in idx]
    ok = np.isfinite(Xv).all(axis=1)
    Xv, yv, spk, ids_v = Xv[ok], yv[ok], spk[ok], [c for c, k in zip(ids_v, ok) if k]

    test_paths = sorted(glob.glob(os.path.join(args.test_dir, "*.wav")))
    print(f"extracting test features ({len(test_paths)} clips)...", flush=True)
    test_rows = collect(test_paths, args.workers)
    Xt = matrix(test_rows)
    ids_t = [r["clip_id"] for r in test_rows]
    okt = np.isfinite(Xt).all(axis=1)
    Xt, ids_t = Xt[okt], [c for c, k in zip(ids_t, okt) if k]

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # speaker-disjoint check first: does this generalise, or memorise sessions?
    speakers = sorted(set(spk))
    rng = np.random.default_rng(0)
    rng.shuffle(speakers)
    held = set(speakers[: len(speakers) // 2])
    tr = np.array([s not in held for s in spk])
    sc = StandardScaler().fit(Xv[tr])
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xv[tr]), yv[tr])
    pv = clf.predict_proba(sc.transform(Xv[~tr]))[:, 1]
    acc = ((pv >= 0.5).astype(int) == yv[~tr]).mean()
    from sklearn.metrics import roc_auc_score
    print(f"held-out speakers: accuracy {acc:.4f}  AUC {roc_auc_score(yv[~tr], pv):.4f}")

    # refit on all of val, then calibrate the cut to the published test prevalence
    sc = StandardScaler().fit(Xv)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xv), yv)
    pt = clf.predict_proba(sc.transform(Xt))[:, 1]
    target = TEST_SYNTHETIC_COUNT * len(pt) / TEST_TOTAL
    thr = float(np.sort(pt)[::-1][int(round(target)) - 1])
    flagged = int((pt >= thr).sum())
    print(f"prevalence-calibrated threshold {thr:.4f} -> {flagged} of {len(pt)} test clips "
          f"synthetic ({flagged/len(pt):.1%}; stated {TEST_SYNTHETIC_COUNT/TEST_TOTAL:.1%})")

    pv_all = clf.predict_proba(sc.transform(Xv))[:, 1]
    val_assign = {c: int(p >= thr) for c, p in zip(ids_v, pv_all)}
    tp = sum(1 for c, y in zip(ids_v, yv) if val_assign[c] == 1 and y == 1)
    fp = sum(1 for c, y in zip(ids_v, yv) if val_assign[c] == 1 and y == 0)
    fn = sum(1 for c, y in zip(ids_v, yv) if val_assign[c] == 0 and y == 1)
    print(f"val at that threshold: precision {tp/max(tp+fp,1):.4f} recall {tp/max(tp+fn,1):.4f}")

    with open(args.out, "w") as fh:
        json.dump({
            "threshold": thr,
            "val_assignments": val_assign,
            "test_assignments": {c: int(p >= thr) for c, p in zip(ids_t, pt)},
            "test_prob": {c: float(p) for c, p in zip(ids_t, pt)},
            "val_prob": {c: float(p) for c, p in zip(ids_v, pv_all)},
            "val_label": {c: int(y) for c, y in zip(ids_v, yv)},
            "test_flagged": flagged,
            "feature_keys": FEATURE_KEYS,
        }, fh)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
