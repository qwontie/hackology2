"""
ABOUTME: Training script for dedicated SSH/remote machine.
ABOUTME: Usage: python train.py
ABOUTME: Edit CONFIG section below to change paths and parameters.
"""
from __future__ import annotations

import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# ============================================================================
# CONFIG — edit this section
# ============================================================================

DATA_DIR      = Path("data")                   # contains train/ and public_test/
OUTPUT_DIR    = Path("output")                 # weights, labels, yaml, predictions
TAXONOMY_PATH = Path("taxonomy.json")

MODEL         = "yolov8m.pt"                   # yolov8n/s/m/l/x or yolo11m.pt etc
EPOCHS        = 50
IMGSZ         = 640
BATCH         = 16
DEVICE        = "0"                            # "0", "0,1", "cpu"
CONFIDENCE    = 0.25

SKIP_TRAIN    = False                          # True = skip training, use existing weights
WEIGHTS       = OUTPUT_DIR / "best.pt"        # used only when SKIP_TRAIN = True


# ============================================================================
# CONVERT COCO → YOLO
# ============================================================================

def convert_coco_to_yolo(
    annotations_path: Path,
    labels_dir: Path,
    taxonomy_path: Path,
) -> tuple[dict[int, int], dict[int, str]]:
    """Convert COCO annotations to YOLO txt format.

    Returns:
        cat_id_to_idx: mapping category_id → yolo class index
        names: mapping yolo index → category name
    """
    labels_dir.mkdir(parents=True, exist_ok=True)

    with open(annotations_path) as f:
        coco = json.load(f)
    with open(taxonomy_path) as f:
        taxonomy = json.load(f)

    cat_ids       = sorted(c["id"] for c in taxonomy["categories"])
    cat_id_to_idx = {cid: idx for idx, cid in enumerate(cat_ids)}
    names         = {
        idx: next(c["name"] for c in taxonomy["categories"] if c["id"] == cid)
        for idx, cid in enumerate(cat_ids)
    }

    img_map     = {img["id"]: img for img in coco["images"]}
    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    converted = 0
    for image_id, anns in anns_by_img.items():
        img_info = img_map[image_id]
        img_w    = img_info["width"]
        img_h    = img_info["height"]
        stem     = Path(img_info["file_name"]).stem

        lines = []
        for ann in anns:
            cat_idx = cat_id_to_idx.get(ann["category_id"])
            if cat_idx is None:
                continue
            x, y, w, h = ann["bbox"]
            x_center   = (x + w / 2) / img_w
            y_center   = (y + h / 2) / img_h
            w_norm     = w / img_w
            h_norm     = h / img_h
            lines.append(f"{cat_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        converted += 1

    print(f"[convert] {converted} label files → {labels_dir}")
    return cat_id_to_idx, names


# ============================================================================
# DATASET YAML
# ============================================================================

def write_dataset_yaml(
    yaml_path: Path,
    images_dir: Path,
    labels_dir: Path,
    names: dict[int, str],
) -> Path:
    lines = [
        f"path: {images_dir.parent.resolve()}",
        f"train: {images_dir.resolve()}",
        f"val: {images_dir.resolve()}",
        f"nc: {len(names)}",
        "names:",
    ]
    for idx, name in sorted(names.items()):
        lines.append(f"  {idx}: {name}")

    yaml_path.write_text("\n".join(lines) + "\n")
    print(f"[yaml] Written: {yaml_path}")
    return yaml_path


# ============================================================================
# TRAIN
# ============================================================================

def train(
    yaml_path: Path,
    output_dir: Path,
    model_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
) -> Path:
    from ultralytics import YOLO

    # parse device
    if "," in device:
        dev = [int(d) for d in device.split(",")]
    else:
        try:
            dev = int(device)
        except ValueError:
            dev = device  # "cpu"

    print(f"[train] model={model_name} epochs={epochs} imgsz={imgsz} batch={batch} device={dev}")

    model   = YOLO(model_name)
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=dev,
        project=str(output_dir / "runs"),
        name="hackology",
        exist_ok=True,
        mosaic=1.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        degrees=5.0,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    final = output_dir / "best.pt"
    shutil.copy(best, final)
    print(f"[train] Best weights: {final}  ({final.stat().st_size / 1024 / 1024:.1f} MB)")
    return final


# ============================================================================
# PREDICT
# ============================================================================

def predict(
    weights: Path,
    test_images_dir: Path,
    test_images_json: Path,
    cat_id_to_idx: dict[int, int],
    output_path: Path,
    confidence: float = 0.25,
) -> None:
    from ultralytics import YOLO

    idx_to_cat_id = {idx: cid for cid, idx in cat_id_to_idx.items()}

    # load image_id map
    if not test_images_json.exists():
        print(f"[predict] WARNING: {test_images_json} not found — image_id will be None")
        filename_to_id = {}
    else:
        with open(test_images_json) as f:
            data = json.load(f)
        imgs = data if isinstance(data, list) else data.get("images", [])
        filename_to_id = {img["file_name"]: img["id"] for img in imgs}
        print(f"[predict] Test images: {len(filename_to_id)}")

    model = YOLO(str(weights))

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = sorted(
        p for p in test_images_dir.iterdir()
        if p.suffix.lower() in image_extensions
    )
    print(f"[predict] Running on {len(image_files)} images (conf={confidence})")

    predictions = []
    skipped = 0

    for img_path in image_files:
        image_id = filename_to_id.get(img_path.name)
        if image_id is None:
            skipped += 1
            continue

        results = model(str(img_path), conf=confidence, verbose=False)

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                cls_id      = int(boxes.cls[i].item())
                category_id = idx_to_cat_id.get(cls_id)
                if category_id is None:
                    continue
                predictions.append({
                    "image_id":    image_id,
                    "category_id": category_id,
                    "bbox":        [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                    "score":       round(float(boxes.conf[i].item()), 4),
                })

    output_path.write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")
    print(f"[predict] {len(predictions)} predictions → {output_path}  (skipped {skipped})")


# ============================================================================
# MAIN
# ============================================================================

def main():
    import torch
    print(f"[info] CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    annotations = DATA_DIR / "train" / "annotations.json"
    images_dir  = DATA_DIR / "train" / "images"
    labels_dir  = OUTPUT_DIR / "labels"
    test_dir    = DATA_DIR / "public_test" / "images"
    test_json   = DATA_DIR / "test_images.json"

    for p in [annotations, images_dir, TAXONOMY_PATH]:
        if not p.exists():
            print(f"[error] Not found: {p}")
            sys.exit(1)

    # 1. convert
    print("\n=== Step 1: Convert COCO → YOLO ===")
    cat_id_to_idx, names = convert_coco_to_yolo(annotations, labels_dir, TAXONOMY_PATH)

    # 2. yaml
    print("\n=== Step 2: Write dataset.yaml ===")
    yaml_path = write_dataset_yaml(
        yaml_path=OUTPUT_DIR / "dataset.yaml",
        images_dir=images_dir,
        labels_dir=labels_dir,
        names=names,
    )

    # 3. train
    if SKIP_TRAIN:
        weights = WEIGHTS
        print(f"\n=== Skipping training, using weights: {weights} ===")
    else:
        print("\n=== Step 3: Train ===")
        weights = train(
            yaml_path=yaml_path,
            output_dir=OUTPUT_DIR,
            model_name=MODEL,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
        )

    # 4. predict
    print("\n=== Step 4: Predict ===")
    predict(
        weights=weights,
        test_images_dir=test_dir,
        test_images_json=test_json,
        cat_id_to_idx=cat_id_to_idx,
        output_path=OUTPUT_DIR / "predictions.json",
        confidence=CONFIDENCE,
    )

    print(f"\n=== Done ===")
    print(f"Weights:     {OUTPUT_DIR / 'best.pt'}")
    print(f"Predictions: {OUTPUT_DIR / 'predictions.json'}")


if __name__ == "__main__":
    main()
