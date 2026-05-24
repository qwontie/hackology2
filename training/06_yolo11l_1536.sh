#!/usr/bin/env bash
# 06: yolo11l @ 1536 — scale-diverse variant of yolo11l_student (trained @ 1280).
# Same architecture, different input scale = mostly-uncorrelated errors on
# small bottles. Used as an *optional* 5th-model swap-in test; final
# 4-model champion did not include it.
# Hyperparams archive: training_metadata/h100_2/y11l_1536_args.yaml
# Produces: runs/detect/yolo11l_1536/weights/best.pt

set -euo pipefail
cd "$(dirname "$0")/.."

DATA="${DATA:-data/data.yaml}"
EPOCHS="${EPOCHS:-25}"
IMGSZ="${IMGSZ:-1536}"
BATCH="${BATCH:-12}"
NAME="${NAME:-yolo11l_1536}"

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
    cache=disk \
    optimizer=AdamW \
    lr0=0.001 \
    lrf=0.01 \
    cos_lr=True \
    patience=15 \
    warmup_epochs=3 \
    weight_decay=0.0005 \
    mosaic=1.0 \
    mixup=0.15 \
    close_mosaic=10 \
    amp=True \
    save_period=5 \
    seed=2026 \
    name="$NAME" \
    project=runs/detect \
    exist_ok=True

echo "=== 06 yolo11l_1536 done: runs/detect/$NAME/weights/best.pt ==="
