#!/usr/bin/env bash
# 02: Student yolo11m + Channel-Wise Distillation (CWD) from the Phase-1 teacher.
# Real source-of-truth: vast_snapshot/v1/train_distill.py (called as a python module,
# not yolo-CLI, because the CWD hook needs both teacher and student model handles).
# Hyperparams archive: training_metadata/h100_1/student_m_1536_cwd_args.yaml
# Produces: runs/detect/student_m_1536_cwd/weights/best.pt   (val mAP50 ~0.90)
#
# REQUIRES: 01_teacher.sh has finished (we read its best.pt as the teacher).
#
# Wall time: ~6h on H100 80GB @ batch=8. Hooked teacher+student forward pass
# at imgsz=1536 is memory-hungry; do not raise batch above 8 without halving imgsz.

set -euo pipefail
cd "$(dirname "$0")/.."

TEACHER_BEST="${TEACHER_BEST:-runs/detect/teacher_x_1536_all/weights/best.pt}"
DATA="${DATA:-data/data.yaml}"
EPOCHS="${EPOCHS:-35}"
IMGSZ="${IMGSZ:-1536}"
BATCH="${BATCH:-8}"
NAME="${NAME:-student_m_1536_cwd}"
SMOKE_FLAG=""

if [[ "${1:-}" == "--smoke" ]]; then EPOCHS=2; NAME="${NAME}_smoke"; SMOKE_FLAG="--smoke"; fi

[[ -f "$TEACHER_BEST" ]] || { echo "missing teacher best.pt at $TEACHER_BEST — run 01_teacher.sh first"; exit 1; }
[[ -f yolo11m.pt ]] || wget -q https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt

# yolo-distiller is a fork of ultralytics with CWD hooks (vast_snapshot/yolo-distiller/).
# vast_snapshot/v1/train_distill.py imports `from ultralytics import YOLO` and
# passes `teacher=teacher.model, distillation_loss="cwd"` to .train().
TEACHER_BEST="$TEACHER_BEST" DATA="$DATA" \
EPOCHS="$EPOCHS" IMGSZ="$IMGSZ" BATCH="$BATCH" \
RUN_NAME="$NAME" \
uv run python vast_snapshot/v1/train_distill.py $SMOKE_FLAG

echo "=== 02 student CWD done: runs/detect/$NAME/weights/best.pt ==="
