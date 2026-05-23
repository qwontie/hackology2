import json
import shutil
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split

def stratified_split(
        annotations_path: Path,
        images_dir: Path,
        output_dir: Path,
        val_size: float = 0.15,
        seed: int = 42,
):
    with open(annotations_path) as f:
        coco = json.load(f)

    # Dominant class per image
    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann["category_id"])

    image_ids = [img["id"] for img in coco["images"]]

    labels = []
    for img_id in image_ids:
        cats = anns_by_img[img_id]
        label = max(set(cats), key=cats.count) if cats else -1
        labels.append(label)

    train_ids, val_ids = train_test_split(
        image_ids,
        test_size=val_size,
        stratify=labels,
        random_state=seed,
    )

    train_ids = set(train_ids)
    val_ids = set(val_ids)

    def build_split(split_ids, split_name):
        img_map = {img["id"]: img for img in coco["images"]}
        split_images = [img_map[i] for i in split_ids]
        split_anns = [a for a in coco["annotations"] if a["image_id"] in split_ids]

        out_img_dir = output_dir / split_name / "images"
        out_ann_dir = output_dir / split_name
        out_img_dir.mkdir(parents=True, exist_ok=True)

        for img in split_images:
            src = images_dir / img["file_name"]
            dst = out_img_dir / img["file_name"]
            if src.exists():
                shutil.copy2(src, dst)

        split_coco = {
            "images": split_images,
            "annotations": split_anns,
            "categories": coco["categories"],
        }
        with open(out_ann_dir / "annotations.json", "w") as f:
            json.dump(split_coco, f, indent=2)

        print(f"{split_name}: {len(split_images)} images, {len(split_anns)} annotations")

    build_split(train_ids, "train")
    build_split(val_ids, "val")


stratified_split(
    annotations_path=Path("data/train/annotations.json"),
    images_dir=Path("data/train/images"),
    output_dir=Path("data"),
    val_size=0.15,
    seed=42,
)