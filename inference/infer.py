"""End-to-end inference for the two Track 1 systems, from a folder of audio to a Codabench submission zip.

  separate (board 1.6168): file-peak router; natural clips -> 6 members x 2 decodes, duration-aware vote;
                           synthetic clips -> 4 specialists, 3-of-4 vote, 30 dB silence gate.
  generic  (no router):    5 members x 2 decodes, one weighted vote on every clip; system_generic.json adds
                           the long-span rule, system_generic_224.json is the board-scored file without it.

usage: python3 inference/infer.py --system configs/system_separate.json --audio-dir DIR --run-dir RUN --name NAME [--gpu 0]
Member decodes are cached under RUN/<member>_<route>/ and reused on a rerun.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")  # same as src/predict.py
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from long_vote import long_fuse  # noqa: E402
from silence_gate import gate  # noqa: E402
from wbf_ensemble import fuse, iou  # noqa: E402


def residuals(run: Path, audio: Path, gpu: str) -> Path:
    """Enhancement residuals (mixture minus U-Net speech) as 16-bit wavs, the input format the residual member was trained on."""
    out = run / "residual"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    with open(run / "residual.log", "w") as log:
        subprocess.run([sys.executable, "-u", "make_residuals.py", "--audio-dir", str(audio), "--out", str(out)],
                       cwd=SRC, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return out


def decode(run: Path, tag: str, ckpt: str, cfg: str, refiner: str | None, audio: Path, gpu: str, bs: int = 16,
           residual_dir: Path | None = None) -> dict:
    out = run / tag
    pred = out / "predictions.jsonl"
    if not pred.exists():
        cmd = [sys.executable, "-u", "predict.py", "--ckpt", str(ROOT / ckpt), "--audio-dir", str(audio),
               "--config", str(ROOT / cfg), "--batch-size", str(bs), "--out", str(out / "pred.zip")]
        if refiner:
            cmd += ["--refiner", str(ROOT / refiner)]
        if residual_dir:
            cmd += ["--residual-dir", str(residual_dir)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        with open(run / f"{tag}.log", "w") as log:
            subprocess.run(cmd, cwd=SRC, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return {(r := json.loads(l))["clip_id"]: [(e["onset"], e["offset"]) for e in r["events"]] for l in open(pred)}


def weighted_vote(spans_by_route, weights, f):
    long_sec = f.get("long_sec", float("inf"))
    cl = []
    for ri, spans in enumerate(spans_by_route):
        for s in spans:
            best, bv = None, (f["iou_long"] if s[1] - s[0] >= long_sec else f["iou"])
            for k in cl:
                v = iou(s, k["box"])
                if v >= bv and ri not in k["src"]:
                    best, bv = k, v
            if best is None:
                cl.append({"box": s, "src": {ri: s}})
            else:
                best["src"][ri] = s
                best["box"] = (float(np.mean([x[0] for x in best["src"].values()])), float(np.mean([x[1] for x in best["src"].values()])))

    def med(k):
        return (float(np.median([v[0] for v in k["src"].values()])), float(np.median([v[1] for v in k["src"].values()])))

    def need(k):
        m = med(k)
        return f["long_keep"] if m[1] - m[0] >= long_sec else f["keep_if_summed_weight_at_least"]
    keep = [k for k in cl if sum(weights[i] for i in k["src"]) >= need(k) - 1e-9] or ([max(cl, key=lambda k: len(k["src"]))] if cl else [])
    return sorted(med(k) for k in keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True); ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--run-dir", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args()
    sysd = json.load(open(a.system)); audio = Path(a.audio_dir).resolve(); run = Path(a.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    wavs = sorted(p for p in audio.rglob("*") if p.suffix.lower() in AUDIO_EXTS)
    clips = [p.stem for p in wavs]
    if not wavs:
        sys.exit(f"no audio under {audio}")
    if len(set(clips)) != len(clips):
        sys.exit(f"duplicate clip names under {audio}")
    path = dict(zip(clips, wavs))
    stamp = run / "audio_dir.txt"
    if stamp.exists() and stamp.read_text().strip() != str(audio):
        sys.exit(f"{run} holds decodes of {stamp.read_text().strip()}; use a fresh --run-dir for {audio}")
    stamp.write_text(str(audio) + "\n")
    out = {}
    if "router" in sysd:
        synthetic = {p.stem for p in wavs if float(np.abs(sf.read(p, dtype="float32")[0]).max()) < 0.995}
        nat_routes = []
        for name, m in sysd["natural"]["members"].items():
            nat_routes.append(decode(run, f"{name}_nat", m["checkpoint"], m["threshold_route_config"], m["refiner"], audio, a.gpu, m["batch_size"]))
            nat_routes.append(decode(run, f"{name}_epnraw", m["checkpoint"], m["raw_route_config"], m["refiner"], audio, a.gpu, m["batch_size"]))
        syn_members = sysd["synthetic"]["members"]
        res = residuals(run, audio, a.gpu) if any(m.get("residual") for m in syn_members.values()) else None
        syn_routes = [decode(run, f"{name}_syn", m["checkpoint"], m["config"], None, audio, a.gpu, m["batch_size"],
                             res if m.get("residual") else None) for name, m in syn_members.items()]
        nf, sf_ = sysd["natural"]["fusion"], sysd["synthetic"]["fusion"]
        for c in clips:
            if c in synthetic:
                ev = [(round(x, 4), round(y, 4)) for x, y in fuse([r[c] for r in syn_routes], sf_["iou"], sf_["votes"], fallback=sf_["fallback"])]
                ev = gate(ev, path[c], sf_["silence_gate_db"]) if ev else ev
            else:
                ev = long_fuse([r[c] for r in nat_routes], nf["votes"], nf["long_sec"], nf["long_votes"], nf["iou_long"])
            out[c] = ev
    else:
        routes, weights = [], []
        for name, m in sysd["members"].items():
            for rt, cfg in (("nat", m["threshold_route_config"]), ("epnraw", m["raw_route_config"])):
                routes.append(decode(run, f"{name}_{rt}", m["checkpoint"], cfg, m["refiner"], audio, a.gpu, m["batch_size"])); weights.append(m["weight"])
        for c in clips:
            out[c] = weighted_vote([r[c] for r in routes], weights, sysd["fusion"])
    lines = [json.dumps({"clip_id": c, "events": [{"onset": round(x, 4), "offset": round(y, 4)} for x, y in sorted(out[c]) if 0 <= x < y]}) for c in clips]
    zp = run / f"{a.name}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("predictions.jsonl", "\n".join(lines) + "\n")
    print(f"wrote {zp}: {len(lines)} clips, {sum(len(out[c]) for c in clips)} events")


if __name__ == "__main__":
    main()
