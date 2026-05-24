"""Convert COCO predictions on public_test to YOLO pseudo-labels for distillation.

Inputs:
  --preds        COCO predictions JSON (e.g. L4_ensemble_heavy.json)
  --images-meta  test_images.json (provides image_id -> file_name + dims + categories)
  --images-dir   directory with public_test images
  --output       output dir (creates images/, labels/, train_pseudo.txt)
  --conf-thr     min confidence for a pred to become a pseudo-label (default 0.65)
  --per-img-cap  max pseudo-labels per image (default 50)

Outputs:
  {output}/images/<file_name>            (symlink to source image)
  {output}/labels/<file_name>.txt        (YOLO: cls cx cy w h, normalized)
  {output}/train_pseudo.txt              (list of pseudo image paths, absolute)
  {output}/pseudo_summary.json           (stats)

Category mapping:
  COCO category_id (1-indexed, 1..369) -> YOLO class (0-indexed, 0..368)
  bottles.yaml uses 0-indexed names, so we subtract 1.

Run example:
  python pseudo_label.py \\
      --preds /tmp/L4_ensemble_heavy.json \\
      --images-meta test_images.json \\
      --images-dir /tmp/public_test/images \\
      --output /tmp/pseudo_data \\
      --conf-thr 0.65
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preds", type=Path, required=True)
    p.add_argument("--images-meta", type=Path, required=True)
    p.add_argument("--images-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--conf-thr", type=float, default=0.65)
    p.add_argument("--per-img-cap", type=int, default=50)
    args = p.parse_args()

    preds = json.loads(args.preds.read_text())
    meta = json.loads(args.images_meta.read_text())
    images = {img["id"]: img for img in meta["images"]}

    out_imgs = args.output / "images"
    out_lbls = args.output / "labels"
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_lbls.mkdir(parents=True, exist_ok=True)

    by_img: dict[int, list[dict]] = defaultdict(list)
    for pred in preds:
        if pred["score"] >= args.conf_thr:
            by_img[pred["image_id"]].append(pred)

    n_kept = 0
    n_imgs = 0
    train_paths: list[str] = []

    for image_id, plist in by_img.items():
        if image_id not in images:
            continue
        info = images[image_id]
        w, h = info["width"], info["height"]
        fname = info["file_name"]

        # cap per image to avoid noisy crowds
        plist.sort(key=lambda x: -x["score"])
        plist = plist[: args.per_img_cap]

        # write label file
        lines: list[str] = []
        for pred in plist:
            x, y, bw, bh = pred["bbox"]
            # COCO xywh (top-left + wh) -> YOLO cxcywh normalized
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            # guard: skip degenerate
            if nw <= 0 or nh <= 0 or nw > 1 or nh > 1:
                continue
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            cls = pred["category_id"] - 1   # COCO 1-indexed -> YOLO 0-indexed
            if cls < 0 or cls > 368:
                continue
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if not lines:
            continue

        # symlink image
        src = args.images_dir / fname
        if not src.exists():
            print(f"  WARN: source image missing: {src}")
            continue
        dst_img = out_imgs / fname
        if dst_img.exists() or dst_img.is_symlink():
            dst_img.unlink()
        dst_img.symlink_to(src.resolve())

        # write label file (YOLO expects label stem == image stem)
        lbl_path = out_lbls / (Path(fname).stem + ".txt")
        lbl_path.write_text("\n".join(lines) + "\n")

        train_paths.append(str(dst_img.resolve()))
        n_kept += len(lines)
        n_imgs += 1

    train_txt = args.output / "train_pseudo.txt"
    train_txt.write_text("\n".join(train_paths) + "\n")

    summary = {
        "input_preds": str(args.preds),
        "conf_thr": args.conf_thr,
        "per_img_cap": args.per_img_cap,
        "total_preds": len(preds),
        "pseudo_labels_kept": n_kept,
        "images_with_labels": n_imgs,
        "output_dir": str(args.output),
        "train_txt": str(train_txt),
    }
    (args.output / "pseudo_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"=== Pseudo-labels generated ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nNext: point bottles_pseudo.yaml at this train_txt and finetune.")


if __name__ == "__main__":
    main()
