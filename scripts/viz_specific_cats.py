"""Render val images for SPECIFIC class IDs with full prediction overlay.

Shows green=GT for the target class, RED=predictions for target class,
YELLOW=predictions for OTHER classes (so we can see WHAT the model confuses it with).

Usage:
    uv run python scripts/viz_specific_cats.py \
        --cat-ids 280 144 157 \
        --val-images data/val/images \
        --val-ann data/val/annotations.json \
        --weights _weights/student_m_1536_cwd_best.pt \
        --out diagnostics/viz_specific
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO
from ultralytics import YOLO


def font_safe(size: int = 16):
    for p in ("/System/Library/Fonts/Helvetica.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_box(draw, bbox_xywh, color: str, label: str, font, width: int = 3) -> None:
    x, y, w, h = bbox_xywh
    x2, y2 = x + w, y + h
    draw.rectangle([x, y, x2, y2], outline=color, width=width)
    tw, th = draw.textbbox((0, 0), label, font=font)[2:]
    draw.rectangle([x, y - th - 2, x + tw + 4, y], fill=color)
    draw.text((x + 2, y - th - 2), label, fill="white", font=font)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat-ids", type=int, nargs="+", required=True)
    ap.add_argument("--val-images", required=True)
    ap.add_argument("--val-ann", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="diagnostics/viz_specific")
    ap.add_argument("--imgsz", type=int, default=1536)
    ap.add_argument("--max-imgs", type=int, default=10, help="max images per class")
    args = ap.parse_args()

    coco_gt = COCO(args.val_ann)
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    cat_id_to_name = {c["id"]: c["name"] for c in cats}
    name_to_coco_id = {c["name"]: c["id"] for c in cats}

    # Find images containing GT for any of the target classes
    image_id_to_filename = {img["id"]: img["file_name"] for img in coco_gt.loadImgs(coco_gt.getImgIds())}
    gt_by_image_cat = defaultdict(lambda: defaultdict(list))  # img_id -> cat_id -> [bbox]
    for a in coco_gt.loadAnns(coco_gt.getAnnIds()):
        gt_by_image_cat[a["image_id"]][a["category_id"]].append(a["bbox"])

    target_set = set(args.cat_ids)
    target_imgs_per_cat = defaultdict(list)
    for img_id, by_cat in gt_by_image_cat.items():
        for cid in target_set:
            if cid in by_cat:
                target_imgs_per_cat[cid].append(img_id)

    val_dir = Path(args.val_images)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    yolo_idx_to_coco_id = {idx: name_to_coco_id.get(nm, idx + 1) for idx, nm in model.names.items()}

    font = font_safe(18)
    summary = {}
    for cid in args.cat_ids:
        cname = cat_id_to_name.get(cid, str(cid))
        safe = "".join(ch if ch.isalnum() else "_" for ch in cname)[:45]
        class_dir = out_root / f"cat{cid:03d}_{safe}"
        class_dir.mkdir(exist_ok=True)
        img_ids = target_imgs_per_cat.get(cid, [])[:args.max_imgs]
        print(f"\n[cat {cid}] {cname}  -> {len(img_ids)} val images")
        rendered = []
        confused_with = defaultdict(int)
        for img_id in img_ids:
            fname = image_id_to_filename[img_id]
            src = val_dir / fname
            if not src.exists():
                continue
            img = Image.open(src).convert("RGB")
            res = model.predict(source=str(src), imgsz=args.imgsz, conf=0.05, iou=0.5,
                                max_det=400, device=0, verbose=False, augment=False)
            draw = ImageDraw.Draw(img)
            # GT for target class (green)
            for bbox in gt_by_image_cat[img_id][cid]:
                draw_box(draw, bbox, "lime", f"GT cat{cid}", font, width=4)
            # Predictions
            boxes = res[0].boxes
            if boxes is not None:
                for b, c, k in zip(boxes.xyxy.cpu().numpy(),
                                    boxes.conf.cpu().numpy(),
                                    boxes.cls.cpu().numpy().astype(int)):
                    coco_c = yolo_idx_to_coco_id.get(int(k), int(k) + 1)
                    bbox = [float(b[0]), float(b[1]), float(b[2]-b[0]), float(b[3]-b[1])]
                    if coco_c == cid:
                        draw_box(draw, bbox, "red", f"P {c:.2f}", font, width=3)
                    else:
                        # Only show "near-GT" wrong-class predictions
                        for gtb in gt_by_image_cat[img_id][cid]:
                            gx, gy, gw, gh = gtb
                            gc_x, gc_y = gx + gw/2, gy + gh/2
                            if (b[0] <= gc_x <= b[2]) and (b[1] <= gc_y <= b[3]):
                                draw_box(draw, bbox, "yellow",
                                         f"!{coco_c} {c:.2f}", font, width=2)
                                confused_with[coco_c] += 1
                                break
            img.save(class_dir / fname, quality=80)
            rendered.append(fname)
        # Summary: top confused-with classes
        top_conf = sorted(confused_with.items(), key=lambda x: -x[1])[:5]
        summary[cid] = {
            "name": cname,
            "n_imgs": len(rendered),
            "confused_with": [{"cat_id": k, "name": cat_id_to_name.get(k, str(k)),
                              "n_hits": v} for k, v in top_conf],
        }
        print(f"   confused with top-5: {[(k, cat_id_to_name.get(k, '?')[:30], v) for k, v in top_conf]}")
        (class_dir / "info.json").write_text(json.dumps(summary[cid], indent=2))

    (out_root / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done] wrote {out_root}")


if __name__ == "__main__":
    main()
