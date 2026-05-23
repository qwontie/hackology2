import json
from pathlib import Path

def coco_to_yolo(annotations_path: Path, images_dir: Path, labels_dir: Path, taxonomy_path: Path):
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
            h_norm = h / img_h
            lines.append(f"{cat_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        label_file = labels_dir / f"{stem}.txt"
        label_file.write_text("\n".join(lines) + "\n")
        converted += 1

    print(f"Skonwertowano {converted} obrazów do {labels_dir}")