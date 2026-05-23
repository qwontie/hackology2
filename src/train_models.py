from ultralytics import YOLO, RTDETR
from pathlib import Path

EPOCHS = 75
IMGSZ  = 1280
BATCH  = 15

def train1():
    model1 = YOLO("yolov8l.pt")
    model1.train(
        data="bottles.yaml",
        device=0,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        seed=42,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        fliplr=0.5,  flipud=0.0,
        mosaic=1.0,  mixup=0.0,
        project="runs", name="model1",
        save=True,
    )

def train2():
    model2 = YOLO("yolov8m.pt")
    model2.train(
        data="bottles.yaml",
        epochs=EPOCHS,
        device=0,
        imgsz=IMGSZ,
        batch=BATCH,
        seed=123,
        hsv_h=0.05, hsv_s=0.9, hsv_v=0.6,
        fliplr=0.5, flipud=0.5,
        mosaic=1.0, mixup=0.2,
        project="runs", name="model2",
        save=True,
    )
def train3():
    # RT-DETR is a transformer detector — needs very different hyperparams than YOLO.
    # First attempt with YOLO-style lr=0.01 + heavy aug collapsed (mAP -> 0 over 5 epochs).
    # Switching to DETR conventions: small AdamW lr, no mosaic/heavy geometric aug, no AMP.
    model3 = RTDETR("rtdetr-l.pt")
    model3.train(
        data="bottles.yaml",
        device=0,
        epochs=50,                # was 75; 50 is enough for DETR with correct lr
        imgsz=IMGSZ,
        batch=BATCH,
        seed=787,
        # --- DETR-friendly optimizer ---
        optimizer="AdamW",
        lr0=1e-4,                 # DETR standard; YOLO default 0.01 collapses transformer
        lrf=0.01,
        weight_decay=1e-4,
        warmup_epochs=2.0,
        amp=False,                # DETR fp16 is unstable in ultralytics
        # --- Aug: minimal (DETR hates mosaic / heavy geometric distortion) ---
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        translate=0.05,
        scale=0.2,
        hsv_h=0.015, hsv_s=0.4, hsv_v=0.3,
        fliplr=0.5,
        flipud=0.0,
        project="runs", name="model3",
        save=True,
    )
def check_results():
    print("Done! Weights are saved")
    for name in ["model1", "model2", "model3"]:
        p = Path(f"runs/{name}/weights/best.pt")
        print(f"  {p}  {'OK' if p.exists() else 'NOT FOUND'}")
