from src.convert_coco_yolo import coco_to_yolo
from pathlib import Path

print("Converting coco to yolo...")
coco_to_yolo(
    annotations_path=Path("data/train_balanced/annotations.json"),
    images_dir=Path("data/train_balanced/images"),
    labels_dir=Path("data/train_balanced/labels"),
    taxonomy_path=Path("taxonomy.json"),
)

coco_to_yolo(
    annotations_path=Path("data/val/annotations.json"),
    images_dir=Path("data/val/images"),
    labels_dir=Path("data/val/labels"),
    taxonomy_path=Path("taxonomy.json"),
)

print("Done!")