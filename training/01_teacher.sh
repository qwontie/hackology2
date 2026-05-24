#!/usr/bin/env bash
# 01: Teacher — yolo11x @ 1536, 40 epochs, AdamW + cos_lr.
# Real source-of-truth hyperparams: training_metadata/h100_1/teacher_x_1536_all_args.yaml
# Produces: runs/detect/teacher_x_1536_all/weights/best.pt   (val mAP50 ~0.92)
#
# Wall time: ~8h on H100 80GB @ batch=12. Halve batch and imgsz=1280 for a
# 24GB consumer card; expect ~12h.

set -euo pipefail
cd "$(dirname "$0")/.."

DATA="${DATA:-data/data.yaml}"
EPOCHS="${EPOCHS:-40}"
IMGSZ="${IMGSZ:-1536}"
BATCH="${BATCH:-12}"
NAME="${NAME:-teacher_x_1536_all}"

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
    optimizer=AdamW \
    lr0=0.001 \
    lrf=0.01 \
    cos_lr=True \
    patience=25 \
    warmup_epochs=3 \
    weight_decay=0.0005 \
    mosaic=1.0 \
    mixup=0.15 \
    copy_paste=0.3 \
    degrees=10 \
    translate=0.1 \
    scale=0.5 \
    fliplr=0.5 \
    close_mosaic=10 \
    amp=True \
    save_period=5 \
    seed=0 \
    name="$NAME" \
    project=runs/detect \
    exist_ok=True

echo "=== 01 teacher done: runs/detect/$NAME/weights/best.pt ==="
