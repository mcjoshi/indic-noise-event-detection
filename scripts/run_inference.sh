#!/bin/bash
# run_inference.sh AUDIO_DIR [generic_224|generic|separate] [GPU]: build the Track 1 submission zip for one of the two systems.
set -euo pipefail
AUDIO=$(readlink -f "$1"); SYSTEM=${2:-separate}; GPU=${3:-0}
[ -d "$AUDIO" ] || { echo "audio folder not found: $1" >&2; exit 1; }
cd "$(dirname "$0")/.."
python3 inference/infer.py --system configs/system_${SYSTEM}.json --audio-dir "$AUDIO" \
  --run-dir outputs/$SYSTEM --name submission_$SYSTEM --gpu $GPU
