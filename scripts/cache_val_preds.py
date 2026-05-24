"""Cache per-model raw val predictions for B6 WBF grid search (codex V4).

Runs each model on val_balanced (or val) at single scale, writes JSON-lines:
  {image_id, bbox: [x,y,w,h], category_id, score, model: <name>}

These files feed scripts/wbf_grid_search.py (next step) which fuses with
varying weights/iou_thr/skip_box_thr and computes mAP against val GT.

Usage on box with weights+val data (H100/H200 idle, or T4):
    python scripts/cache_val_preds.py \\
        --weights _weights/student_m_1536_cwd_best.pt _weights/teacher_x_1536_all_best.pt \\
                  _weights/yolo11l_student_best.pt _weights/yolov8l_train1_best.pt \\
        --val-images data/val/images \\
        --val-anno data/val/annotations.json \\
        --imgsz 1536 \\
        --out-dir /tmp/val_preds_cache
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--val-images", required=True)
    ap.add_argument("--val-anno", required=True)
    ap.add_argument("--imgsz", type=int, default=1536)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.90,
                    help="Ultralytics NMS IoU. Codex: 0.85 serious candidate, 0.90 worth sweep. "
                         "Cache at 0.90 (most permissive) so WBF sees max candidates.")
    ap.add_argument("--max-det", type=int, default=600)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out-dir", default="/tmp/val_preds_cache")
    args = ap.parse_args()

    from ultralytics import YOLO  # type: ignore

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    anno = json.loads(Path(args.val_anno).read_text())
    fname_to_id = {im["file_name"]: im["id"] for im in anno["images"]}
    img_dir = Path(args.val_images)
    img_paths = sorted(p for p in img_dir.iterdir() if p.name in fname_to_id and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"[load] {len(img_paths)} val images, {len(args.weights)} models")

    # YOLO indices are 0-based; categories in val anno are 1-based usually.
    # We emit RAW category_id from the model output (0-based); calling grid
    # search must translate using the same convention as predict_v2.
    for wp in args.weights:
        name = Path(wp).stem
        out_file = out / f"{name}__imgsz{args.imgsz}_iou{int(args.iou*100)}.jsonl"
        if out_file.exists():
            print(f"[skip] {out_file} already exists")
            continue
        print(f"[load model] {wp}")
        model = YOLO(wp)
        t0 = time.time()
        with out_file.open("w") as f:
            for i, p in enumerate(img_paths):
                res = model.predict(source=str(p), imgsz=args.imgsz, conf=args.conf,
                                    iou=args.iou, max_det=args.max_det, device=args.device,
                                    verbose=False, half=True)
                boxes = res[0].boxes
                if boxes is None:
                    continue
                xyxy = boxes.xyxy.cpu().tolist()
                cls = boxes.cls.cpu().tolist()
                cf = boxes.conf.cpu().tolist()
                img_id = fname_to_id[p.name]
                for j in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[j]
                    f.write(json.dumps({
                        "image_id": img_id,
                        "bbox": [round(x1, 2), round(y1, 2),
                                 round(x2 - x1, 2), round(y2 - y1, 2)],
                        "cls_idx": int(cls[j]),
                        "score": round(float(cf[j]), 4),
                    }) + "\n")
                if (i + 1) % 50 == 0:
                    rate = (i + 1) / (time.time() - t0)
                    eta = (len(img_paths) - i - 1) / rate
                    print(f"  [{name}] {i+1}/{len(img_paths)} {rate:.2f}img/s eta={eta:.0f}s",
                          flush=True)
        sz = out_file.stat().st_size / 1024 / 1024
        print(f"[done] {name} -> {out_file} ({sz:.1f}MB) {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
