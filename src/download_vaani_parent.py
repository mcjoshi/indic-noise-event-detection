"""Download the untranscribed Vaani shards we used (gated dataset ARTPARK-IISc/Vaani, needs HF_TOKEN) and link them
into one flat folder for prep_data.py.

usage: python3 src/download_vaani_parent.py --shards configs/data/vaani_parent_shards_round1.txt --flat data/parent_raw_flat
"""
import argparse
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

from paths import DATA

ap = argparse.ArgumentParser()
ap.add_argument("--shards", required=True, help="text file of shard paths inside the dataset repo")
ap.add_argument("--raw", default=f"{DATA}/parent_raw")
ap.add_argument("--flat", required=True)
a = ap.parse_args()

flat = Path(a.flat)
flat.mkdir(parents=True, exist_ok=True)
shards = open(a.shards).read().split()
for i, s in enumerate(shards):
    p = Path(hf_hub_download("ARTPARK-IISc/Vaani", s, repo_type="dataset", token=os.environ["HF_TOKEN"], local_dir=a.raw))
    link = flat / s.removeprefix("audio/").replace("/", "_")
    if not link.exists():
        link.symlink_to(p.resolve())
    print(f"{i + 1}/{len(shards)} {s}", flush=True)
