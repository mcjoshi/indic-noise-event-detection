"""Run a trained model over the evaluation audio and build the Codabench submission.

Produces a zip containing a single `predictions.jsonl` **at the zip root** - Codabench
scoring programs generally reject a submission whose files sit inside a folder.

Every clip in the evaluation set must appear exactly once.  If a clip list is supplied
the output is emitted in that order and any clip the model did not see is written with
an empty event list rather than silently dropped, since a missing line is a malformed
submission while an empty prediction is merely a bad one.
"""

from __future__ import annotations

import argparse
import os
import pickle
import json
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from dataset import SR, n_frames_for
from boundary_refiner import refine_spans
from epn_eval import extend_long_spans
from epn_head import epn_decode
from clip_policy import apply as policy_apply
from model import decode, load_checkpoint, snap_to_boundaries, tta_forward

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")


def find_audio(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTS)


def load_clip_ids(path: Path) -> list[str]:
    """Accept a plain id-per-line list, or a jsonl/json carrying clip_id fields."""
    text = path.read_text().strip()
    if path.suffix == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("clips", data.get("clip_ids", []))
        return [d["clip_id"] if isinstance(d, dict) else str(d) for d in data]
    ids = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            ids.append(json.loads(line)["clip_id"])
        else:
            ids.append(line)
    return ids


@torch.no_grad()
def predict(
    model, paths: list[Path], device: str, cfg: dict, batch_size: int = 16,
    snap: bool = True, refiner=None, residual_dir: Path | None = None, clip_policy=None
) -> tuple[dict[str, list[tuple[float, float]]], dict[str, float]]:
    events, durations = {}, {}
    for i in range(0, len(paths), batch_size):
        chunk = paths[i : i + batch_size]
        wavs, ids, durs, ress = [], [], [], []
        for p in chunk:
            x, sr = sf.read(p, dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            if sr != SR:
                raise ValueError(f"{p} is {sr} Hz, expected {SR}")
            wavs.append(x)
            ids.append(p.stem)
            durs.append(len(x) / SR)
            if residual_dir is not None:
                r, _ = sf.read(residual_dir / f"{p.stem}.wav", dtype="float32")
                ress.append((r.mean(axis=1) if r.ndim > 1 else r)[: len(x)])

        n_max = max(len(w) for w in wavs)
        batch = np.zeros((len(wavs), n_max), dtype=np.float32)
        for j, w in enumerate(wavs):
            batch[j, : len(w)] = w
        n_fr = n_frames_for(n_max)
        res_t = None
        if ress:
            rb = np.zeros((len(ress), n_max), dtype=np.float32)
            for j, r in enumerate(ress):
                rb[j, : len(r)] = r
            res_t = torch.from_numpy(rb).to(device)

        tensor = torch.from_numpy(batch).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = tta_forward(model, tensor, n_fr, res_t) if os.environ.get("TTA_SHIFT") == "1" else model(tensor, n_fr, res_t)
        probs = torch.sigmoid(out[0].float()).cpu().numpy()
        p_on = torch.sigmoid(out[2].float()).cpu().numpy()
        p_off = torch.sigmoid(out[3].float()).cpu().numpy()
        dist = out[5].float().cpu().numpy()

        # snap_sec is a decode-config key consumed here, not a decode() argument; the
        # tuner searches it with 0 meaning "boundary snapping off".  An "epn" block
        # switches the whole decode to greedy proposal selection.
        decode_cfg = dict(cfg)
        snap_sec = decode_cfg.pop("snap_sec", 0.0)
        epn_cfg = dict(decode_cfg.pop("epn", None) or {})
        epn_refine = epn_cfg.pop("refine", True)
        epn_extend = epn_cfg.pop("extend_min", None)

        for j, cid in enumerate(ids):
            keep = n_frames_for(len(wavs[j]))
            if epn_cfg:
                spans = epn_decode(probs[j, :keep], dist[j, :keep, 0], dist[j, :keep, 1],
                                   duration=durs[j], **epn_cfg)
                if epn_extend:
                    spans = extend_long_spans(
                        spans, decode(probs[j, :keep], duration=durs[j], **decode_cfg), epn_extend)
                if refiner is not None and epn_refine:
                    spans = refine_spans(spans, probs[j, :keep], p_on[j, :keep],
                                         p_off[j, :keep], refiner, device, durs[j])
                events[cid] = spans
                durations[cid] = durs[j]
                continue
            cfg_j = decode_cfg
            if clip_policy is not None:
                cfg_j = dict(decode_cfg, threshold=policy_apply(clip_policy, probs[j, :keep], durs[j]))
            spans = decode(probs[j, :keep], duration=durs[j], **cfg_j)
            if snap and snap_sec:
                spans = snap_to_boundaries(
                    spans, p_on[j, :keep], p_off[j, :keep], search_sec=snap_sec
                )
            if refiner is not None:
                # proposal-conditioned correction of each endpoint, applied last so it
                # sees the spans the decoder actually produced
                spans = refine_spans(spans, probs[j, :keep], p_on[j, :keep],
                                     p_off[j, :keep], refiner, device, durs[j])
            events[cid] = spans
            durations[cid] = durs[j]

        if (i // batch_size) % 20 == 0:
            print(f"  {min(i + batch_size, len(paths))}/{len(paths)} clips", flush=True)
    return events, durations


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--config", required=True, help="json from tune_postproc.py")
    ap.add_argument("--out", required=True, help="output zip path")
    ap.add_argument("--clip-list", default=None,
                    help="optional file of required clip ids, one per line or jsonl")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--no-snap", action="store_true",
                    help="disable boundary-head refinement")
    ap.add_argument("--refiner", default=None,
                    help="trained BoundaryRefiner checkpoint")
    ap.add_argument("--clip-policy", default=None, help="per-clip threshold policy pickle (clip_policy.py)")
    ap.add_argument("--residual-dir", default=None,
                    help="directory of enhancement residual wavs named like the audio (two-channel models)")
    args = ap.parse_args()

    device = "cuda"
    model, ckpt = load_checkpoint(args.ckpt, device, use_ema=not args.no_ema)
    print(f"loaded {args.ckpt} (val Combined {ckpt.get('combined', float('nan')):.4f})")

    cfg = json.loads(Path(args.config).read_text())
    cfg = cfg.get("config", cfg)
    print(f"decode config: {json.dumps(cfg)}")

    paths = find_audio(Path(args.audio_dir))
    if not paths:
        raise SystemExit(f"no audio found under {args.audio_dir}")
    print(f"{len(paths)} audio files")

    refiner = None
    if args.refiner:
        from boundary_refiner import load_refiner
        refiner = load_refiner(args.refiner, device)
        refiner.eval()
        print(f"boundary refiner: {args.refiner}")

    events, durations = predict(model, paths, device, cfg, args.batch_size,
                                snap=not args.no_snap, refiner=refiner,
                                residual_dir=Path(args.residual_dir) if args.residual_dir else None,
                                clip_policy=pickle.load(open(args.clip_policy, "rb")) if args.clip_policy else None)

    if args.clip_list:
        required = load_clip_ids(Path(args.clip_list))
        missing = [c for c in required if c not in events]
        req = set(required)
        extra = [c for c in events if c not in req]
        if missing:
            print(f"WARNING: {len(missing)} required clips had no audio; "
                  f"writing empty predictions for them (e.g. {missing[:3]})")
        if extra:
            print(f"WARNING: dropping {len(extra)} predicted clips not in the clip list")
        order = required
    else:
        order = sorted(events)

    out_zip = Path(args.out)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_zip.parent / "predictions.jsonl"

    n_events = 0
    with open(jsonl_path, "w") as fh:
        for cid in order:
            evs = events.get(cid, [])
            n_events += len(evs)
            fh.write(json.dumps({
                "clip_id": cid,
                "events": [{"onset": float(a), "offset": float(b)} for a, b in evs],
            }) + "\n")

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(jsonl_path, arcname="predictions.jsonl")   # at the zip root

    empty = sum(1 for cid in order if not events.get(cid))
    print(f"\nwrote {out_zip}")
    print(f"  {len(order)} clips, {n_events} events "
          f"({n_events / max(len(order), 1):.2f} per clip), {empty} clips with no events")
    print(f"  uncompressed jsonl kept at {jsonl_path}")


if __name__ == "__main__":
    main()
