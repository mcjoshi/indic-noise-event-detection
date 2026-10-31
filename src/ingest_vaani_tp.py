"""Ingest the transcribed Vaani release (ARTPARK-IISc/Vaani-transcription-part, external data, disclosed) for
transcript-conditioned labelling. Shard by shard: download, keep clips whose transcript carries at least one noise-event
tag, drop every clip that is a competition clip or comes from a recording behind any competition test or validation
clip, write the wav and a manifest row, delete the shard. Resumable (done shards listed in the log).
usage: python3 ingest_vaani_tp.py [--root data/vaani_tp] [--max-gb 150]
"""
from __future__ import annotations
from paths import DATA, PRETRAINED, WORK  # noqa: F401

import argparse
import json
import os
import re
import shutil

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download

REPO = "ARTPARK-IISc/Vaani-transcription-part"
NONEV = {"noise", "pause", "static noise", "talking", "unintelligible", "people talking", "inaudible"}
TAG = re.compile(r"\[([^\]]+)\]|<([^/>][^>]*)>")


def norm_transcript(t: str) -> str:
    """Map the release's tag variants onto the competition's (`[horn]` -> `<horn> </horn>`, `_` -> space)."""
    t = re.sub(r"<(/?)([^>]+)>", lambda m: f"<{m.group(1)}{m.group(2).replace('_', ' ').strip().lower()}>", t or "")
    def br(m):
        name = m.group(1).replace("_", " ").strip().lower()
        return f"[{name}]" if name in ("breathing", "inhaling", "lip smacking", "unintelligible", "inaudible", "noise", "talking") else f"<{name}> </{name}>"
    return re.sub(r"\[([^\]]+)\]", br, t)


def n_events(t: str) -> int:
    return sum(1 for m in TAG.finditer(t) if (m.group(1) or m.group(2)).strip() not in NONEV)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=f"{DATA}/vaani_tp")
    ap.add_argument("--max-gb", type=float, default=150.0, help="per wav directory; past it, spill to --spill-dir")
    ap.add_argument("--spill-dir", default=f"{DATA}/vaani_tp_wav2")
    a = ap.parse_args()
    tok = os.environ["HF_TOKEN"]
    ex = json.load(open(f"{a.root}/exclude.json")); clips = set(ex["clips"]); blocked = set(ex["blocked_recordings"])
    log = f"{a.root}/ingest.log"; done = set(l.split("\t")[1] for l in open(log) if l.startswith("DONE")) if os.path.exists(log) else set()
    files = sorted(s.rfilename for s in HfApi(token=tok).dataset_info(REPO).siblings if s.rfilename.endswith(".parquet"))
    wavdir = f"{a.root}/wav"; os.makedirs(wavdir, exist_ok=True)
    man = open(f"{a.root}/manifest.jsonl", "a")
    for f in files:
        if f in done:
            continue
        if wavdir != a.spill_dir and sum(e.stat().st_size for e in os.scandir(wavdir)) / 1e9 > a.max_gb:
            wavdir = a.spill_dir; os.makedirs(wavdir, exist_ok=True); print(f"SPILL to {wavdir}", flush=True)
        p = hf_hub_download(REPO, f, repo_type="dataset", token=tok, local_dir=f"{a.root}/tmp")
        t = pq.read_table(p, columns=["audio", "language", "state", "district", "transcript"])
        kept = skipped = tagless = 0
        for r in t.to_pylist():
            cid = os.path.basename(r["audio"]["path"])[:-4]
            rec = re.sub(r"_\d+_\d+$", "", cid)
            if cid in clips or rec in blocked:
                skipped += 1; continue
            tr = norm_transcript(r["transcript"])
            if n_events(tr) == 0:
                tagless += 1; continue
            out = f"{wavdir}/{cid}.wav"
            with open(out, "wb") as fh:
                fh.write(r["audio"]["bytes"])
            dur = sf.info(out).duration
            man.write(json.dumps({"clip_id": cid, "wav": out, "duration": round(dur, 3), "language": r["language"], "state": r["state"],
                                  "district": r["district"], "transcript": tr, "n_event_tags": n_events(tr)}) + "\n")
            kept += 1
        man.flush(); os.remove(p)
        with open(log, "a") as fh:
            fh.write(f"DONE\t{f}\tkept {kept}\ttagless {tagless}\texcluded {skipped}\n")
        print(f"{f}: kept {kept} tagless {tagless} excluded {skipped}", flush=True)
    shutil.rmtree(f"{a.root}/tmp", ignore_errors=True)
    print("INGEST_DONE", flush=True)


if __name__ == "__main__":
    main()
