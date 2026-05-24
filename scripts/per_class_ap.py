"""Compute per-class AP@0.5 on val using a single model.

Usage:
    uv run python scripts/per_class_ap.py \
        --weights _weights/student_m_1536_cwd_best.pt \
        --data-root /workspace/hackology2/data \
        --out diagnostics/per_class_student.json \
        --imgsz 1536

Outputs JSON list, sorted ascending by AP50:
    [{"cat_id": 31, "name": "...", "ap50": 0.0,
      "train_n": 12, "val_n": 5,
      "worst_val_images": [{"file": "abc.jpg", "gt_n": 3}, ...]}, ...]

Also writes diagnostics/weak_class_targets.json (top-K worst with worst-image
filenames) for the visualization step.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO


def run_val_inference(weights: str, val_images_dir: Path, imgsz: int,
                       device: int = 0, conf: float = 0.001):
    """Run YOLO on every val image one-by-one (avoids 'too many open files')."""
    import resource
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 32768), hard))
    except Exception:
        pass

    model = YOLO(weights)
    img_paths = sorted(val_images_dir.glob("*.jpg")) + sorted(val_images_dir.glob("*.png"))
    print(f"[infer] {len(img_paths)} val images, imgsz={imgsz}, conf={conf}")

    pairs = []
    for i, p in enumerate(img_paths):
        # one image per predict call → ultralytics opens only one file at a time
        res = model.predict(
            source=str(p),
            imgsz=imgsz, conf=conf, iou=0.5, max_det=400,
            device=device, augment=False, verbose=False, stream=False,
        )
        pairs.append((p, res[0]))
        if (i + 1) % 100 == 0:
            print(f"[infer] {i + 1}/{len(img_paths)}")
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-root", required=True,
                    help="dir containing val/{images,annotations.json} and train_balanced/annotations.json")
    ap.add_argument("--out", default="diagnostics/per_class_ap.json")
    ap.add_argument("--imgsz", type=int, default=1536)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--topk-weak", type=int, default=30,
                    help="how many worst classes to surface for visualization step")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    val_ann_path = data_root / "val" / "annotations.json"
    val_imgs_dir = data_root / "val" / "images"
    train_ann_path = data_root / "train_balanced" / "annotations.json"

    print(f"[load] val annotations: {val_ann_path}")
    coco_gt = COCO(str(val_ann_path))
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    cat_id_to_name = {c["id"]: c["name"] for c in cats}
    # YOLO outputs 0-indexed class indices. COCO annotations are 1-indexed
    # category_ids in our dataset (verified: cat_id 1 == YAML idx 0). Build the
    # mapping from name (YAML is source of truth) so we don't assume offset.
    name_to_coco_id = {c["name"]: c["id"] for c in cats}

    # File name -> image_id from val annotations
    filename_to_image_id = {img["file_name"]: img["id"] for img in coco_gt.loadImgs(coco_gt.getImgIds())}

    # --- 1. Train sample count per category
    print(f"[load] train annotations: {train_ann_path}")
    with open(train_ann_path) as f:
        train_ann = json.load(f)
    train_count = defaultdict(int)
    for a in train_ann["annotations"]:
        train_count[a["category_id"]] += 1

    val_count = defaultdict(int)
    for a in coco_gt.loadAnns(coco_gt.getAnnIds()):
        val_count[a["category_id"]] += 1

    # --- 2. Run inference, convert to COCO detections
    print(f"[infer] running {args.weights}")
    img_results = run_val_inference(args.weights, val_imgs_dir,
                                     imgsz=args.imgsz, device=args.device)

    # Build YOLO-idx -> COCO-cat-id map from the loaded model's name table
    # (ultralytics exposes model.names as {idx: name}). Falls back to +1 if
    # name lookup fails.
    yolo_idx_to_coco_id = {}

    detections = []
    missing_files = 0
    for img_path, res in img_results:
        fname = img_path.name
        if fname not in filename_to_image_id:
            missing_files += 1
            continue
        image_id = filename_to_image_id[fname]
        if not yolo_idx_to_coco_id and getattr(res, "names", None):
            for idx, nm in res.names.items():
                if nm in name_to_coco_id:
                    yolo_idx_to_coco_id[int(idx)] = name_to_coco_id[nm]
                else:
                    yolo_idx_to_coco_id[int(idx)] = int(idx) + 1  # fallback
            print(f"[map] built yolo-idx -> coco-cat-id map, {len(yolo_idx_to_coco_id)} entries; "
                  f"sample: 0 -> {yolo_idx_to_coco_id.get(0)}  1 -> {yolo_idx_to_coco_id.get(1)}")
        if res.boxes is None or len(res.boxes) == 0:
            continue
        boxes_xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clses = res.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), c, k in zip(boxes_xyxy, confs, clses):
            w, h = float(x2 - x1), float(y2 - y1)
            if w < 1 or h < 1:
                continue
            coco_cat = yolo_idx_to_coco_id.get(int(k), int(k) + 1)
            detections.append({
                "image_id": int(image_id),
                "category_id": int(coco_cat),
                "bbox": [float(x1), float(y1), w, h],
                "score": float(c),
            })
    print(f"[infer] {len(detections)} detections (missing_files={missing_files})")

    # --- 3. COCO eval (per-class via cocoeval.stats with categoryId loop)
    coco_dt = coco_gt.loadRes(detections)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # precision shape: [T, R, K, A, M]
    #   T=10 IoU thresholds (.50:.05:.95), R=101 recall, K=#cats, A=4 areas, M=3 max_dets
    # AP@0.5 = mean over recall, taking IoU index 0, area=all (0), max_dets=last (-1)
    precision = coco_eval.eval["precision"]  # type: ignore
    cat_ids_sorted = sorted(cat_id_to_name.keys())
    ap50_by_cat = {}
    for ki, cat_id in enumerate(cat_ids_sorted):
        p = precision[0, :, ki, 0, -1]  # IoU=0.5
        valid = p[p > -1]
        ap50_by_cat[cat_id] = float(np.mean(valid)) if valid.size > 0 else 0.0

    # --- 4. Build per-class report
    report = []
    for cat_id in cat_ids_sorted:
        report.append({
            "cat_id": cat_id,
            "name": cat_id_to_name.get(cat_id, str(cat_id)),
            "ap50": ap50_by_cat[cat_id],
            "train_n": train_count.get(cat_id, 0),
            "val_n": val_count.get(cat_id, 0),
        })
    report.sort(key=lambda x: x["ap50"])  # worst first

    # --- 5. For weak classes, find worst val images (most GT, no/few predictions)
    weak = report[:args.topk_weak]
    img_pred_count = defaultdict(lambda: defaultdict(int))  # img_id -> cat_id -> #preds
    for d in detections:
        img_pred_count[d["image_id"]][d["category_id"]] += 1

    img_gt_count = defaultdict(lambda: defaultdict(int))
    for a in coco_gt.loadAnns(coco_gt.getAnnIds()):
        img_gt_count[a["image_id"]][a["category_id"]] += 1

    image_id_to_filename = {img["id"]: img["file_name"] for img in coco_gt.loadImgs(coco_gt.getImgIds())}

    for r in weak:
        cid = r["cat_id"]
        # images that contain GT for this class
        gt_imgs = [(img_id, img_gt_count[img_id][cid])
                   for img_id in img_gt_count if img_gt_count[img_id][cid] > 0]
        # rank by gt_n minus preds_n (most missed first)
        scored = sorted(gt_imgs,
                        key=lambda x: (-x[1], img_pred_count[x[0]].get(cid, 0)))
        r["worst_val_images"] = [
            {"image_id": int(img_id),
             "file": image_id_to_filename[img_id],
             "gt_n": int(img_gt_count[img_id][cid]),
             "pred_n": int(img_pred_count[img_id].get(cid, 0))}
            for img_id, _ in scored[:5]
        ]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "weights": args.weights,
            "imgsz": args.imgsz,
            "mAP50_all": float(coco_eval.stats[1]),
            "mAP50_95_all": float(coco_eval.stats[0]),
            "n_classes": len(report),
            "n_classes_ap0": sum(1 for r in report if r["ap50"] == 0.0),
            "n_classes_ap_below_0.2": sum(1 for r in report if r["ap50"] < 0.2),
            "per_class": report,
        }, f, indent=2)
    print(f"\n[done] wrote {args.out}")
    print(f"[summary] mAP50={coco_eval.stats[1]:.4f}  "
          f"AP=0 classes: {sum(1 for r in report if r['ap50'] == 0.0)}, "
          f"AP<0.2 classes: {sum(1 for r in report if r['ap50'] < 0.2)}")
    print(f"[summary] worst 10:")
    for r in report[:10]:
        print(f"  cat {r['cat_id']:>3}  AP50={r['ap50']:.4f}  "
              f"train_n={r['train_n']:>4}  val_n={r['val_n']:>3}  {r['name']}")


if __name__ == "__main__":
    main()
