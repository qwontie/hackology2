"""Probe submission predict: run YOLO inference on public_test, output COCO json."""
import json
import sys
from pathlib import Path
from ultralytics import YOLO

MODEL = "/workspace/v1/runs/teacher_x_1536_all/weights/best.pt"
IMG_DIR = Path("/workspace/v1/data3/data/public_test/images")
TEST_JSON = Path("/workspace/v1/data3/data/public_test/test_images.json")
TAXONOMY = Path("/workspace/v1/taxonomy.json")
OUT = Path("/workspace/v1/submissions/predictions_epoch5_probe.json")
OUT.parent.mkdir(exist_ok=True)

CONF = 0.001
IMGSZ = 1536
MAX_DET = 300

with open(TEST_JSON) as f:
    test = json.load(f)
with open(TAXONOMY) as f:
    tax = json.load(f)

# YOLO idx -> hackathon category_id (sorted by id, matches convert_data.py)
cats = sorted(tax["categories"], key=lambda c: c["id"])
idx_to_cat = {i: c["id"] for i, c in enumerate(cats)}
fname_to_id = {im["file_name"]: im["id"] for im in test["images"]}

print(f"Test images: {len(fname_to_id)}, categories: {len(idx_to_cat)}", file=sys.stderr)

model = YOLO(MODEL)
predictions = []

# Build sorted list of test image paths
img_paths = []
for fname in fname_to_id:
    p = IMG_DIR / fname
    if not p.exists():
        print(f"MISSING: {fname}", file=sys.stderr)
        continue
    img_paths.append(p)

print(f"Found {len(img_paths)} test image files on disk", file=sys.stderr)

# Batch inference
results = model.predict(
    source=[str(p) for p in img_paths],
    imgsz=IMGSZ,
    conf=CONF,
    max_det=MAX_DET,
    device=0,
    verbose=False,
    stream=True,
    half=True,
)

for img_path, result in zip(img_paths, results):
    image_id = fname_to_id[img_path.name]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        continue
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        w = x2 - x1
        h = y2 - y1
        cls_id = int(boxes.cls[i].item())
        cat_id = idx_to_cat.get(cls_id)
        if cat_id is None:
            continue
        predictions.append({
            "image_id": image_id,
            "category_id": cat_id,
            "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
            "score": round(float(boxes.conf[i].item()), 4),
        })

with open(OUT, "w") as f:
    json.dump(predictions, f)

print(f"Wrote {len(predictions)} predictions to {OUT}", file=sys.stderr)
print(f"Size: {OUT.stat().st_size / 1024 / 1024:.2f} MB", file=sys.stderr)
