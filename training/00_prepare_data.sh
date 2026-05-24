#!/usr/bin/env bash
# Prepare the YOLO-formatted dataset under $DATA_ROOT/data/.
#
# Expects organiser-provided COCO data (images + annotations.json) to be
# downloaded already, OR uses vast_snapshot/v1/download_data.sh if you're on a
# vast.ai box with the bundle in the staging area.
#
# Outputs:
#   $DATA_ROOT/data/data.yaml       (dataset yaml for ultralytics)
#   $DATA_ROOT/data/train/{images,labels}/
#   $DATA_ROOT/data/val/{images,labels}/
#   $DATA_ROOT/data/train_balanced/  (optional, K-shot capped; needed for *_cb runs)
#
# Usage:
#   bash training/00_prepare_data.sh                      # use ./data/ as root
#   DATA_ROOT=/workspace/hackology2 bash training/00_prepare_data.sh

set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-$(pwd)}"
K_CAP="${K_CAP:-12}"   # per-class cap for class-balanced split

echo "[00] DATA_ROOT=$DATA_ROOT"

# 1. (optional) pull raw COCO bundle if running on a vast image we already have
if [[ ! -d "$DATA_ROOT/data/train/images" && -x "vast_snapshot/v1/download_data.sh" ]]; then
    echo "[00] running vast_snapshot/v1/download_data.sh"
    bash vast_snapshot/v1/download_data.sh
fi

# 2. Convert COCO -> YOLO format + write data.yaml
python3 prepare_dataset.py --data-root "$DATA_ROOT/data"

# 3. Class-balanced trainlist (only used by *_cb runs)
python3 scripts/build_balanced_trainlist.py \
    --data-root "$DATA_ROOT/data" \
    --split train_balanced \
    --K "$K_CAP" \
    --out "$DATA_ROOT/data/train_balanced.txt"

echo "[00] done. data.yaml -> $DATA_ROOT/data/data.yaml"
ls -la "$DATA_ROOT/data/"
