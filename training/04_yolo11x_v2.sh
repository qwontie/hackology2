#!/usr/bin/env bash
# 04: yolo11x_v2 — fresh-seed yolo11x for ensemble diversity (+0.0074 LB).
# IDENTICAL recipe to the teacher except seed=42 and imgsz=1280 (faster).
# This is the model that took us from public 0.8042 (3-model) to 0.8116 (4-model).
# Hyperparams archive: training_metadata/h100_1/y11x_v2_args.yaml
# Produces: runs/detect/yolo11x_v2/weights/best.pt    (val mAP50 ~0.73)

set -euo pipefail
cd "$(dirname "$0")/.."

DATA="${DATA:-data/data.yaml}"
EPOCHS="${EPOCHS:-30}"
IMGSZ="${IMGSZ:-1280}"
BATCH="${BATCH:-14}"
NAME="${NAME:-yolo11x_v2}"

if [[ "${1:-}" == "--smoke" ]]; then EPOCHS=2; NAME="${NAME}_smoke"; fi

[[ -f yolo11x.pt ]] || wget -q https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt

uv run yolo detect train \
    model=yolo11x.pt \
    data="$DATA" \
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
    mixup=0.15 \
    copy_paste=0.0 \
    degrees=0 \
    translate=0.1 \
    scale=0.5 \
    fliplr=0.5 \
    close_mosaic=10 \
    amp=True \
    save_period=5 \
    seed=42 \
    name="$NAME" \
    project=runs/detect \
    exist_ok=True

echo "=== 04 yolo11x_v2 done: runs/detect/$NAME/weights/best.pt ==="
