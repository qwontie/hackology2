"""Train yolo11l student on full data (DPR_MIR + SIDG + synth).

Per codex review: this is the strongest realistic addition to our ensemble:
  - Bigger than yolo11m student (val mAP probably 0.91-0.93)
  - Different from teammates' yolov8l (modern architecture)
  - Trained with proven YOLO recipe, so low risk vs alternative experiments

Run on a free H100 80GB box with hackology2 repo cloned and uv sync'd:
    cd /workspace/hackology2 && uv run python scripts/train_yolo11l.py

Outputs:
    runs/yolo11l_student/weights/best.pt   <-- this is what gets uploaded to GH Release
    runs/yolo11l_student/weights/last.pt
    runs/yolo11l_student/results.csv
"""
from __future__ import annotations

from ultralytics import YOLO


def main() -> None:
    model = YOLO("yolo11l.pt")   # COCO-pretrained backbone
    model.train(
        data="bottles.yaml",
        device=0,
        # --- schedule ---
        epochs=45,
        imgsz=1536,
        batch=16,                # tune if OOM (H100 80GB should handle this)
        seed=123,
        # --- optimizer ---
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,
        weight_decay=5e-4,
        # --- augmentation (YOLO-proven recipe) ---
        mosaic=1.0,
        mixup=0.05,
        copy_paste=0.2,
        close_mosaic=10,         # disable mosaic in last 10 epochs for stable convergence
        degrees=8.0,
        translate=0.08,
        scale=0.45,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        # --- output ---
        project="runs",
        name="yolo11l_student",
        save=True,
        save_period=-1,          # only best + last
        plots=True,
        verbose=True,
        amp=True,                # YOLO is fine with AMP (unlike DETR)
        patience=15,             # early stop if no val improvement in 15 epochs
    )


if __name__ == "__main__":
    main()
