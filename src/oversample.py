import json
import shutil
import random
from pathlib import Path
from collections import defaultdict


def oversample_rare_classes(
        annotations_path: Path,
        images_dir: Path,
        output_dir: Path,
        min_annotations: int = 50,
        seed: int = 42,
):
    random.seed(seed)

    with open(annotations_path) as f:
        coco = json.load(f)

    # Annotations count
    class_counts = defaultdict(int)
    for ann in coco["annotations"]:
        class_counts[ann["category_id"]] += 1

    # Images by class
    imgs_by_class = defaultdict(set)
    for ann in coco["annotations"]:
        imgs_by_class[ann["category_id"]].add(ann["image_id"])

    img_map = {img["id"]: img for img in coco["images"]}

    new_images = list(coco["images"])
    new_annotations = list(coco["annotations"])

    next_img_id = max(img["id"] for img in coco["images"]) + 1
    next_ann_id = max(ann["id"] for ann in coco["annotations"]) + 1

    out_images_dir = output_dir / "images"
    out_images_dir.mkdir(parents=True, exist_ok=True)

    # Copy original images
    for img in coco["images"]:
        src = images_dir / img["file_name"]
        dst = out_images_dir / img["file_name"]
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    for cat_id, count in class_counts.items():
        if count >= min_annotations:
            continue

        times = (min_annotations // count)  # times
        img_ids = list(imgs_by_class[cat_id])

        for i in range(times):
            img_id = random.choice(img_ids)
            orig_img = img_map[img_id]

            stem = Path(orig_img["file_name"]).stem
            ext = Path(orig_img["file_name"]).suffix
            new_filename = f"{stem}_dup{i}_{cat_id}{ext}"

            # Copy image
            src = images_dir / orig_img["file_name"]
            dst = out_images_dir / new_filename
            if src.exists():
                shutil.copy2(src, dst)

            # New image record
            new_img = {**orig_img, "id": next_img_id, "file_name": new_filename}
            new_images.append(new_img)

            # Copy annotations
            for ann in coco["annotations"]:
                if ann["image_id"] == img_id:
                    new_ann = {**ann, "id": next_ann_id, "image_id": next_img_id}
                    new_annotations.append(new_ann)
                    next_ann_id += 1

            next_img_id += 1

    new_coco = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": coco["categories"],
    }

    out_ann = output_dir / "annotations.json"
    with open(out_ann, "w") as f:
        json.dump(new_coco, f, indent=2)

    print(f"Original images: {len(coco['images'])}")
    print(f"After oversampling: {len(new_images)}")