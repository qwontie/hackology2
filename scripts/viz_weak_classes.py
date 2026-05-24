"""Render collages for weak classes (GT in green, predictions in red).

Reads diagnostics/per_class_student.json (output of per_class_ap.py), takes
top-K worst classes, and for each writes diagnostics/weak_class_viz/cat_XX/
containing up to 5 annotated val images.

Usage (run locally after pulling diagnostics/ + val images):
    uv run python scripts/viz_weak_classes.py \
        --diag diagnostics/per_class_student.json \
        --val-images data/val/images \
        --val-ann data/val/annotations.json \
        --weights _weights/student_m_1536_cwd_best.pt \
        --out diagnostics/weak_class_viz \
        --topk 30
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO
from ultralytics import YOLO


def draw_box(draw: ImageDraw.ImageDraw, bbox_xywh, color: str, label: str,
             width: int = 3) -> None:
    x, y, w, h = bbox_xywh
    x2, y2 = x + w, y + h
    draw.rectangle([x, y, x2, y2], outline=color, width=width)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except Exception:
        font = ImageFont.load_default()
    tw, th = draw.textbbox((0, 0), label, font=font)[2:]
    draw.rectangle([x, y - th - 2, x + tw + 4, y], fill=color)
    draw.text((x + 2, y - th - 2), label, fill="white", font=font)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", required=True)
    ap.add_argument("--val-images", required=True)
    ap.add_argument("--val-ann", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="diagnostics/weak_class_viz")
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=1536)
    ap.add_argument("--per-class-imgs", type=int, default=5)
    args = ap.parse_args()

    diag = json.loads(Path(args.diag).read_text())
    coco_gt = COCO(args.val_ann)
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    cat_id_to_name = {c["id"]: c["name"] for c in cats}
    name_to_coco_id = {c["name"]: c["id"] for c in cats}

    weak = diag["per_class"][:args.topk]
    target_images = set()
    for r in weak:
        for wi in r.get("worst_val_images", []):
            target_images.add(wi["file"])
    print(f"[plan] {len(weak)} weak classes × ~{args.per_class_imgs} images = "
          f"{len(target_images)} unique images to re-infer")

    val_images_dir = Path(args.val_images)
    model = YOLO(args.weights)

    # batch-infer just on the union of needed images
    img_paths = [val_images_dir / f for f in target_images if (val_images_dir / f).exists()]
    print(f"[infer] {len(img_paths)} images on disk")
    results = model.predict(
        source=[str(p) for p in img_paths],
        imgsz=args.imgsz, conf=0.001, iou=0.5, max_det=400,
        device=0, augment=False, verbose=False, stream=False,
    )
    # Build yolo-idx -> coco-cat-id map from model name table (matches per_class_ap.py)
    yolo_idx_to_coco_id = {}
    if results and getattr(results[0], "names", None):
        for idx, nm in results[0].names.items():
            yolo_idx_to_coco_id[int(idx)] = name_to_coco_id.get(nm, int(idx) + 1)

    preds_by_file = {}
    for p, r in zip(img_paths, results):
        if r.boxes is None:
            preds_by_file[p.name] = []
            continue
        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clses = r.boxes.cls.cpu().numpy().astype(int)
        preds_by_file[p.name] = [
            {"bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
             "score": float(c),
             "cat_id": yolo_idx_to_coco_id.get(int(k), int(k) + 1)}
            for (x1, y1, x2, y2), c, k in zip(boxes_xyxy, confs, clses)
        ]

    # GT by image
    gt_by_image = defaultdict(list)  # img_id -> list of {cat_id, bbox}
    for a in coco_gt.loadAnns(coco_gt.getAnnIds()):
        gt_by_image[a["image_id"]].append({"cat_id": a["category_id"], "bbox": a["bbox"]})
    filename_to_image_id = {img["file_name"]: img["id"]
                            for img in coco_gt.loadImgs(coco_gt.getImgIds())}

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    for r in weak:
        cid = r["cat_id"]
        cname = cat_id_to_name.get(cid, str(cid))
        safe_name = "".join(ch if ch.isalnum() else "_" for ch in cname)[:40]
        class_dir = out_root / f"cat{cid:03d}_{safe_name}_ap{int(r['ap50']*100):03d}"
        class_dir.mkdir(parents=True, exist_ok=True)
        info = {
            "cat_id": cid,
            "name": cname,
            "ap50": r["ap50"],
            "train_n": r["train_n"],
            "val_n": r["val_n"],
            "imgs": [],
        }
        for wi in r.get("worst_val_images", [])[:args.per_class_imgs]:
            fname = wi["file"]
            src = val_images_dir / fname
            if not src.exists():
                continue
            img = Image.open(src).convert("RGB")
            draw = ImageDraw.Draw(img)

            img_id = filename_to_image_id.get(fname)
            for g in gt_by_image.get(img_id, []):
                if g["cat_id"] == cid:
                    draw_box(draw, g["bbox"], "lime", f"GT:{cid}")
            # predictions for this class (all confidences)
            for p in preds_by_file.get(fname, []):
                if p["cat_id"] == cid:
                    draw_box(draw, p["bbox"], "red",
                             f"P{cid}:{p['score']:.2f}")

            out_path = class_dir / fname
            img.save(out_path, quality=80)
            info["imgs"].append({"file": fname, "gt_n": wi["gt_n"],
                                 "pred_n": wi["pred_n"]})

        (class_dir / "info.json").write_text(json.dumps(info, indent=2))

    print(f"[done] wrote {len(weak)} class folders under {out_root}")
    print(f"       open them visually — green = GT, red = student pred at conf≥0.001")


if __name__ == "__main__":
    main()
