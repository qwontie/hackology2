#!/usr/bin/env bash
# 05: yolov8x with class-balanced sampling — cross-family diversity.
# Real source-of-truth: scripts/launch_yolo11x_cb_h200.sh (which trains yolo11x_cb);
# this is the v8 variant we kept for the final 5-model probe.
# Hyperparams archive: training_metadata/h100_2/yolov8x_cb_args.yaml
# Produces: runs/detect/yolov8x_cb/weights/best.pt    (val mAP50 ~0.73)
#
# Requires: training/00_prepare_data.sh has built data/train_balanced.txt

set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-$(pwd)/data}"
DATA="${DATA:-data/data.yaml}"
EPOCHS="${EPOCHS:-30}"
IMGSZ="${IMGSZ:-1536}"
BATCH="${BATCH:-12}"
NAME="${NAME:-yolov8x_cb}"
TRAIN_LIST="${TRAIN_LIST:-$DATA_ROOT/train_balanced.txt}"

if [[ "${1:-}" == "--smoke" ]]; then EPOCHS=2; NAME="${NAME}_smoke"; fi

[[ -f yolov8x.pt ]] || wget -q https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x.pt
[[ -f "$TRAIN_LIST" ]] || { echo "missing $TRAIN_LIST — run 00_prepare_data.sh first"; exit 1; }

# Materialise a CB data yaml that points train: to the balanced list
CB_YAML="bottles_cb.yaml"
python3 - <<PY
import yaml, pathlib
src = yaml.safe_load(open("$DATA"))
src["train"] = "$TRAIN_LIST"
pathlib.Path("$CB_YAML").write_text(yaml.safe_dump(src, sort_keys=False))
PY

uv run yolo detect train \
    model=yolov8x.pt \
    data="$CB_YAML" \
    epochs="$EPOCHS" \
    imgsz="$IMGSZ" \
    batch="$BATCH" \
    device=0 \
    workers=8 \
    cache=disk \
    optimizer=AdamW \
    lr0=0.001 \
    lrf=0.01 \
    cos_lr=True \
    patience=20 \
    warmup_epochs=3 \
    weight_decay=0.0005 \
    mosaic=1.0 \
    mixup=0.2 \
    close_mosaic=10 \
    amp=True \
    save_period=5 \
    seed=787 \
    name="$NAME" \
    project=runs/detect \
    exist_ok=True

echo "=== 05 yolov8x_cb done: runs/detect/$NAME/weights/best.pt ==="
