#!/usr/bin/env bash
# 03: yolo11l student — diversity hero (+0.017 LB by itself in WBF).
# Trained from COCO-pretrained yolo11l, NOT distilled. Different arch family
# than yolo11m/x produces uncorrelated errors that WBF eats up.
# Real source-of-truth: scripts/train_yolo11l.py
# Hyperparams archive: training_metadata/h100_2/yolo11l_student{,2}_args.yaml
# Produces: runs/detect/yolo11l_student/weights/best.pt    (val mAP50 ~0.70)

set -euo pipefail
cd "$(dirname "$0")/.."

DATA="${DATA:-data/data.yaml}"
EPOCHS="${EPOCHS:-45}"
IMGSZ="${IMGSZ:-1536}"
BATCH="${BATCH:-16}"
NAME="${NAME:-yolo11l_student}"

if [[ "${1:-}" == "--smoke" ]]; then EPOCHS=2; NAME="${NAME}_smoke"; fi

[[ -f yolo11l.pt ]] || wget -q https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt

uv run yolo detect train \
    model=yolo11l.pt \
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
    patience=15 \
    warmup_epochs=3 \
    weight_decay=0.0005 \
    mosaic=1.0 \
    mixup=0.05 \
    copy_paste=0.2 \
    degrees=8 \
    translate=0.08 \
    scale=0.45 \
    fliplr=0.5 \
    close_mosaic=10 \
    amp=True \
    seed=123 \
    name="$NAME" \
    project=runs/detect \
    exist_ok=True

echo "=== 03 yolo11l student done: runs/detect/$NAME/weights/best.pt ==="
