import json
from pathlib import Path
def coco_to_yolo(annotations_path: Path, images_dir: Path, labels_dir: Path, taxonomy_path: Path):
    """Konwertuj anotacje COCO na format YOLO (txt per obraz).

    Każdy plik .txt zawiera linie: class_idx x_center y_center width height
    (znormalizowane do [0, 1]).
    """
    labels_dir.mkdir(parents=True, exist_ok=True)

    with open(annotations_path) as f:
        coco = json.load(f)
    with open(taxonomy_path) as f:
        taxonomy = json.load(f)

    cat_ids = sorted(c["id"] for c in taxonomy["categories"])
    cat_id_to_idx = {cid: idx for idx, cid in enumerate(cat_ids)}

    img_map = {img["id"]: img for img in coco["images"]}

    from collections import defaultdict
    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    converted = 0
    for image_id, anns in anns_by_img.items():
        img_info = img_map[image_id]
        img_w = img_info["width"]
        img_h = img_info["height"]
        stem = Path(img_info["file_name"]).stem

        lines = []
        for ann in anns:
            cat_idx = cat_id_to_idx.get(ann["category_id"])
            if cat_idx is None:
                continue
            x, y, w, h = ann["bbox"]  # COCO: [x, y, w, h] (top-left)
            # YOLO: [x_center, y_center, w, h] (normalized)
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_hfrom ultralytics import YOLO
from pathlib import Path



EPOCHS = 100
IMGSZ  = 640
BATCH  = 16   

model1 = YOLO("yolov8s.pt")
model1.train(
    data="bottles.yaml",
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    seed=42,
    fl_gamma=2.0,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    fliplr=0.5,  flipud=0.0,
    mosaic=1.0,  mixup=0.0,
    project="runs", name="model1",
    save=True,
)
model2 = YOLO("yolov8m.pt")
model2.train(
    data="bottles.yaml",
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    seed=123,
    fl_gamma=2.0,
    hsv_h=0.05, hsv_s=0.9, hsv_v=0.6,
    fliplr=0.5, flipud=0.5,
    mosaic=1.0, mixup=0.2,
    project="runs", name="model2",
    save=True,
)
model3 = YOLO("rtdetr-l.pt")
model3.train(
    data="bottles.yaml",
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    seed=777,
    fl_gamma=2.0,ą
    degrees=15.0, translate=0.2,
    scale=0.6,    shear=5.0,
    mosaic=0.5,
    project="runs", name="model3",
    save=True,
)
print("Готово! Веса сохранены:")
for name in ["model1", "model2", "model3"]:
    p = Path(f"runs/{name}/weights/best.pt")
    print(f"  {p}  {'✓' if p.exists() else '✗ НЕ НАЙДЕН'}")ą
            lines.append(f"{cat_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        label_file = labels_dir / f"{stem}.txt"
        label_file.write_text("\n".join(lines) + "\n")
        converted += 1

    print(f"Skonwertowano {converted} obrazów do {labels_dir}")

coco_to_yolo(
    annotations_path=Path("data/train/annotations.json"),
    images_dir=Path("data/train/images"),
    labels_dir=Path("data/train/labels"),
    taxonomy_path=Path("taxonomy.json"),
)