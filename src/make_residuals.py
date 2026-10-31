"""Noise-only view of every clip: mixture minus the Track 2 U-Net's speech estimate.

Writes one residual wav per input wav, mirroring the relative path under --out, so the
detector can take (mixture, residual) as a two-channel input.
"""
from __future__ import annotations
from paths import DATA, PRETRAINED, WORK  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from track2_unet import TorchSTFT, UNetMasker

SR = 16000


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unet", default=str(PRETRAINED / "unet_v4.pt"))
    ap.add_argument("--manifest", default=None, help="jsonl with wav paths relative to --audio-root")
    ap.add_argument("--audio-dir", default=None, help="or a directory of wavs (test set)")
    ap.add_argument("--audio-root", default=f"{DATA}/prepared")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th clip starting at i")
    ap.add_argument("--absolute-only", action="store_true", help="only manifest entries with absolute wav paths")
    args = ap.parse_args()
    device = "cuda"
    ck = torch.load(args.unet, map_location="cpu")
    ch = tuple(int(c) for c in ck["args"]["channels"].split(","))
    model = UNetMasker(floor=ck.get("floor", 0.05), ch=ch).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    stft = TorchSTFT().to(device)

    if args.manifest:
        recs = [json.loads(l) for l in open(args.manifest) if l.strip()]
        from dataset import rel_wav
        items = [(Path(args.audio_root) / r["wav"], Path(args.out) / rel_wav(r["wav"])) for r in recs]
        if args.absolute_only:
            items = [(s, d) for s, d in items if Path(str(s)).is_absolute() or str(s).startswith("/")]
    else:
        items = [(p, Path(args.out) / f"{p.stem}.wav") for p in sorted(Path(args.audio_dir).rglob("*"))
                 if p.suffix.lower() in (".wav", ".flac", ".mp3", ".ogg", ".m4a")]
    i, n = (int(x) for x in args.shard.split("/"))
    items = items[i::n]
    seen = set()
    done = 0
    for src, dst in items:
        if dst in seen or dst.exists():
            continue
        seen.add(dst)
        x, sr = sf.read(src, dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        assert sr == SR
        t = torch.from_numpy(x).unsqueeze(0).to(device)
        spec = stft.stft(t)
        lm = torch.log1p(spec.abs()).transpose(1, 2).unsqueeze(1)
        g = model(lm)
        g01 = ((g - model.floor) / (1.0 - model.floor)).clamp(0, 1)
        speech = stft.istft(spec * g01.transpose(1, 2), length=t.shape[-1])
        res = (t - speech).squeeze(0).cpu().numpy().astype(np.float32)
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(dst, res, SR, subtype="PCM_16")
        done += 1
        if done % 2000 == 0:
            print(f"  {done}/{len(items)}", flush=True)
    print(f"wrote {done} residuals under {args.out}")


if __name__ == "__main__":
    main()
