"""Download the released checkpoints and pretrained encoders into checkpoints/ and pretrained/.

usage: python3 scripts/download_assets.py [--repo manishjoshiai/indic-noise-event-detection] [--train]
--train also fetches HuBERT-large and the training manifests, expanding "$DATA/" in them to the data directory.
"""
import argparse
import gzip
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

ROOT = Path(__file__).resolve().parent.parent
PRETRAINED = Path(os.environ.get("VAANI_PRETRAINED", ROOT / "pretrained"))
DATA = Path(os.environ.get("VAANI_DATA", ROOT / "data"))

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="manishjoshiai/indic-noise-event-detection", help="Hugging Face model repo holding checkpoints/, pretrained/ and data/")
ap.add_argument("--train", action="store_true", help="also fetch what training needs")
a = ap.parse_args()

snapshot_download(a.repo, local_dir=ROOT, allow_patterns=["checkpoints/*"])
PRETRAINED.mkdir(parents=True, exist_ok=True)
for f in list_repo_files(a.repo):
    dst = PRETRAINED / Path(f).name
    if f.startswith("pretrained/") and not dst.exists():
        shutil.copy(hf_hub_download(a.repo, f), dst)
snapshot_download("openai/whisper-medium", local_dir=PRETRAINED / "whisper_medium",
                  allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"])
snapshot_download("microsoft/wavlm-base-plus", local_dir=PRETRAINED / "wavlm_base_plus",
                  allow_patterns=["config.json", "preprocessor_config.json", "pytorch_model.bin"])

if a.train:
    snapshot_download("facebook/hubert-large-ll60k", local_dir=PRETRAINED / "hubert_large",
                      allow_patterns=["config.json", "preprocessor_config.json", "pytorch_model.bin"])
    snapshot_download(a.repo, local_dir=ROOT, allow_patterns=["data/*"])
    for gz in sorted((ROOT / "data").rglob("*.jsonl.gz")):
        dst = DATA / gz.relative_to(ROOT / "data").with_suffix("")
        dst.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(gz, "rt") as f, open(dst, "w") as g:
            for line in f:
                g.write(line.replace('"$DATA/', f'"{DATA.resolve()}/'))
        print("manifest", dst)

print("assets in place: checkpoints/ and", PRETRAINED)
