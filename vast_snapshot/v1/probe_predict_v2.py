"""Probe predict v2: memory-efficient single-image loop."""
import json, sys, time, gc
from pathlib import Path
import torch
from ultralytics import YOLO

MODEL = "/workspace/v1/runs/teacher_x_1536_all/weights/epoch5.pt"
IMG_DIR = Path("/workspace/v1/data3/data/public_test/images")
TEST_JSON = Path("/workspace/v1/data3/data/public_test/test_images.json")
TAXONOMY = Path("/workspace/v1/taxonomy.json")
OUT = Path("/workspace/v1/submissions/predictions_epoch5.json")
OUT.parent.mkdir(exist_ok=True)

CONF = 0.001
IMGSZ = 1536
MAX_DET = 300

with open(TEST_JSON) as f: test = json.load(f)
with open(TAXONOMY) as f: tax = json.load(f)

cats = sorted(tax["categories"], key=lambda c: c["id"])
idx_to_cat = {i: c["id"] for i, c in enumerate(cats)}
fname_to_id = {im["file_name"]: im["id"] for im in test["images"]}

img_paths = []
for fname in fname_to_id:
    p = IMG_DIR / fname
    if p.exists(): img_paths.append(p)

print(f"Loading model...", file=sys.stderr, flush=True)
model = YOLO(MODEL)

predictions = []
t0 = time.time()

for i, img_path in enumerate(img_paths):
    image_id = fname_to_id[img_path.name]
    # Single-image inference with explicit cleanup
    results = model.predict(
        source=str(img_path),
        imgsz=IMGSZ,
        conf=CONF,
        max_det=MAX_DET,
        device=0,
        verbose=False,
        half=True,
    )
    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        # Move to CPU immediately
        xyxy = boxes.xyxy.cpu().tolist()
        cls = boxes.cls.cpu().tolist()
        conf = boxes.conf.cpu().tolist()
        for j in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[j]
            cat_id = idx_to_cat.get(int(cls[j]))
            if cat_id is None: continue
            predictions.append({
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [round(x1, 2), round(y1, 2), round(x2-x1, 2), round(y2-y1, 2)],
                "score": round(float(conf[j]), 4),
            })
    del results
    if (i + 1) % 50 == 0:
        torch.cuda.empty_cache()
        gc.collect()
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (len(img_paths) - i - 1) / rate
        mem = torch.cuda.memory_allocated() / 1024**3
        print(f"  [{i+1}/{len(img_paths)}] {rate:.2f} img/s, eta={eta:.0f}s, preds={len(predictions)}, gpu={mem:.1f}GB", file=sys.stderr, flush=True)

with open(OUT, "w") as f:
    json.dump(predictions, f)

print(f"\nDONE {time.time()-t0:.0f}s, {len(predictions)} preds, {OUT.stat().st_size/1024/1024:.2f}MB", file=sys.stderr)
