#!/bin/bash
# train_member.sh RUN [GPU]: train one model from its recorded configuration, then fit its decode configs:
# threshold decode -> boundary refiner -> natural (threshold) route -> EPN route -> EPN raw route.
# RUN is a file name (without .json) under configs/members/, configs/lineage/ (ancestors) or configs/donors/.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=$1; GPU=${2:-0}
CFG=configs/members/$RUN.json; [ -f "$CFG" ] || CFG=configs/lineage/$RUN.json; [ -f "$CFG" ] || CFG=configs/donors/$RUN.json
export CUDA_VISIBLE_DEVICES=$GPU
W=work
python3 src/train.py $(python3 src/replay_args.py "$CFG")
python3 src/tune_postproc.py --ckpt $W/crnn_$RUN/best.pt --val data/splits/val_mixed.jsonl --audio-root data/prepared \
  --out $W/postproc_$RUN.json --cache $W/val_mixed_probs_$RUN.pkl --rounds 4
python3 src/train_refiner.py --ckpt $W/crnn_$RUN/best.pt --train-manifest data/splits/train_gold_v7.jsonl --audio-root data/prepared \
  --val-cache $W/val_mixed_probs_$RUN.pkl --config $W/postproc_$RUN.json --out $W/refiner_$RUN.pt --limit 12000 --epochs 40
python3 src/tune_natural_route.py --cache $W/val_mixed_probs_$RUN.pkl --refiner $W/refiner_$RUN.pt \
  --out $W/postproc_${RUN}_natural.json --rounds 3
python3 src/tune_epn.py --ckpt $W/crnn_$RUN/best.pt --config $W/postproc_${RUN}_natural.json \
  --refiner $W/refiner_$RUN.pt --cache $W/val_epn_outputs_$RUN.pkl --out $W/postproc_${RUN}_epnroute.json
python3 - "$RUN" <<'PY'
import json, sys
run = sys.argv[1]
cfg = json.load(open(f"work/postproc_{run}_epnroute.json"))
cfg["config"]["epn"] = {k: v for k, v in cfg["config"]["epn"].items() if k != "extend_min"}
cfg["config"]["epn"]["refine"] = False
json.dump(cfg, open(f"work/postproc_{run}_epnraw.json", "w"), indent=1)
PY
