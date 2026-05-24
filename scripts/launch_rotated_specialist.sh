#!/usr/bin/env bash
# Rotated-bottle specialist train (per Gemini's worth_it=yes + Артем's idea).
#
# Background: Gemini found ~3+ failure modes from lying/rotated bottles:
#   - 15e2343d (15× ChiReg_12Yo_KAR duplicates on shelf)
#   - 04992f30, 82b23f32 (lying bottles fragmented into multi-boxes)
#   - General: standard NMS can't handle long horizontal aspect bottles.
#
# We don't have OBB GT, so this is NOT a true rotation-aware model. Instead:
#   - Filter train images that ALREADY contain at least one bbox with aspect<0.7
#     (model will see lying/wide bottles in normal training format)
#   - Heavy rotation augmentation in train transforms
#   - yolo11m for speed (2-3× faster than 11x at imgsz=1280, fine for specialist)
#   - 20 epochs (specialist, not from-scratch)
#   - Specialist gets WBF weight 0.3-0.5 in final ensemble — supports main models
#     on lying-bottle cases without overpowering them on standing-bottle cases.
#
# Usage on H100 #2 AFTER y8x_cb finishes (free GPU at ~0%):
#   scp -P <port> scripts/launch_rotated_specialist.sh root@<host>:/workspace/hackology2/scripts/
#   ssh -p <port> root@<host>
#   cd /workspace/hackology2 && tmux new -s rot -d 'bash scripts/launch_rotated_specialist.sh 2>&1 | tee /root/rot_train.log'

set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/workspace/hackology2/data}"
SPLIT="${SPLIT:-train_balanced}"
ASPECT_MAX="${ASPECT_MAX:-0.7}"     # only include images with at least one bbox of w/h < this
EPOCHS="${EPOCHS:-20}"
IMGSZ="${IMGSZ:-1280}"
BATCH="${BATCH:-24}"
NAME="${NAME:-rot_specialist}"

[[ -d "$DATA_ROOT/$SPLIT/labels" ]] || { echo "missing $DATA_ROOT/$SPLIT/labels"; exit 1; }
[[ -f yolo11m.pt ]] || { echo "fetching yolo11m.pt..."; wget -q https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt; }

# 1. Build rotated-image list (aspect < $ASPECT_MAX = wide/lying)
ROT_LIST="$DATA_ROOT/${SPLIT}_rotated.txt"
python3 - <<PY
import pathlib
root = pathlib.Path("$DATA_ROOT/$SPLIT")
labels = root / "labels"
images = root / "images"
ASP = float("$ASPECT_MAX")
kept = 0
total = 0
with open("$ROT_LIST", "w") as f:
    for lp in labels.glob("*.txt"):
        total += 1
        has_rot = False
        for line in lp.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                _c, _x, _y, w, h = parts[:5]
                w, h = float(w), float(h)
                if h > 0 and (w / h) < ASP:
                    has_rot = True
                    break
            except ValueError:
                continue
        if has_rot:
            # try jpg, fallback png
            stem = lp.stem
            for ext in ("jpg", "png", "jpeg"):
                p = images / f"{stem}.{ext}"
                if p.exists():
                    f.write(str(p) + "\n")
                    kept += 1
                    break
print(f"[scan] {kept}/{total} imgs have bbox aspect<{ASP}")
PY

# 2. Write yaml for specialist (val unchanged)
ROT_YAML="bottles_rot.yaml"
python3 - <<PY
import yaml, pathlib
src = yaml.safe_load(open("bottles.yaml"))
src["train"] = "$ROT_LIST"
src["path"] = "$DATA_ROOT"
pathlib.Path("$ROT_YAML").write_text(yaml.safe_dump(src, sort_keys=False))
print("wrote $ROT_YAML")
PY

# 3. Train.  Heavy rotation aug + standard mosaic/mixup.
#    degrees=15 -> rotate up to ±15° (above this is rare in real images)
#    flipud=0.3 -> upside-down flip 30% of time (lying bottles can be either orientation)
yolo detect train \
    model=yolo11m.pt \
    data="$ROT_YAML" \
    epochs="$EPOCHS" \
    imgsz="$IMGSZ" \
    batch="$BATCH" \
    device=0 \
    patience=15 \
    cache=disk \
    workers=8 \
    mosaic=1.0 \
    mixup=0.15 \
    degrees=15 \
    flipud=0.3 \
    fliplr=0.5 \
    close_mosaic=8 \
    optimizer=AdamW \
    lr0=0.0015 \
    cos_lr=True \
    seed=911 \
    name="$NAME" \
    project=runs/detect \
    exist_ok=True \
    save_period=5

echo
echo "=== DONE === best at runs/detect/$NAME/weights/best.pt ==="
ls -la "runs/detect/$NAME/weights/"
