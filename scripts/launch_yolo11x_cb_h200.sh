#!/usr/bin/env bash
# Launch yolo11x class-balanced training on an H200 NVL box.
# Codex V3 recommended H200 bet (zero integration: drops into predict_v2 WBF).
#
# Usage on a fresh H200 vast.ai box (after git clone + uv sync OR pip install ultralytics):
#   scp -P <port> scripts/build_balanced_trainlist.py scripts/launch_yolo11x_cb_h200.sh \
#       root@<host>:/workspace/hackology2/scripts/
#   ssh -p <port> root@<host>
#   cd /workspace/hackology2
#   tmux new -s y11x_cb -d 'bash scripts/launch_yolo11x_cb_h200.sh 2>&1 | tee y11x_cb_train.log'
#   tmux attach -t y11x_cb   # detach with Ctrl-b d
#
# Expected: 30 epochs @ 1536 batch=12 on H200 ~3-4h. Final best.pt at:
#   runs/detect/yolo11x_cb/weights/best.pt
# Then upload to GH Release as `yolo11x_cb_best.pt` and add 'yolo11x_cb' to WEIGHT_ALIASES.

set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

DATA_ROOT="${DATA_ROOT:-/workspace/hackology2/data}"
SPLIT="${SPLIT:-train_balanced}"
K="${K:-6}"
# YOLO 0-indexed sibling boost targets: 279=cat280 Pas 1L, 143=cat144 ChiReg12Yo 1L, 156=cat157 ChiReg12Yo KAR
BOOST="${BOOST:-279 143 156}"
BOOST_FACTOR="${BOOST_FACTOR:-3}"
EPOCHS="${EPOCHS:-30}"
IMGSZ="${IMGSZ:-1536}"
BATCH="${BATCH:-12}"
NAME="${NAME:-yolo11x_cb}"

# 0. Pre-flight
command -v yolo >/dev/null || { echo "ultralytics CLI not found; run: pip install ultralytics"; exit 1; }
[[ -f yolo11x.pt ]] || { echo "fetching yolo11x.pt..."; wget -q https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt; }
[[ -d "$DATA_ROOT/$SPLIT/labels" ]] || { echo "missing $DATA_ROOT/$SPLIT/labels"; exit 1; }

# 1. Build balanced train list
TRAIN_LIST="$DATA_ROOT/${SPLIT}_cb.txt"
python3 scripts/build_balanced_trainlist.py \
    --data-root "$DATA_ROOT" \
    --split "$SPLIT" \
    --K "$K" \
    --boost-classes $BOOST \
    --boost-factor "$BOOST_FACTOR" \
    --out "$TRAIN_LIST"

# 2. Write a temp YAML pointing train to the list (val unchanged)
CB_YAML="bottles_cb.yaml"
python3 - <<PY
import yaml, pathlib
src = yaml.safe_load(open("bottles.yaml"))
src["train"] = "$TRAIN_LIST"
src["path"] = "$DATA_ROOT"
pathlib.Path("$CB_YAML").write_text(yaml.safe_dump(src, sort_keys=False))
print("wrote $CB_YAML")
PY

# 3. Train.  close_mosaic=10 → last 10 ep without mosaic to stabilize; mixup=0.2 helps sibling discrimination.
yolo detect train \
    model=yolo11x.pt \
    data="$CB_YAML" \
    epochs="$EPOCHS" \
    imgsz="$IMGSZ" \
    batch="$BATCH" \
    device=0 \
    patience=20 \
    cache=disk \
    workers=8 \
    mosaic=1.0 \
    mixup=0.2 \
    close_mosaic=10 \
    optimizer=AdamW \
    lr0=0.001 \
    cos_lr=True \
    seed=787 \
    name="$NAME" \
    project=runs/detect \
    exist_ok=True \
    save_period=5

echo
echo "=== DONE === best at runs/detect/$NAME/weights/best.pt ==="
ls -la "runs/detect/$NAME/weights/"
