"""Phase 2: Teacher fine-tune on DPR_MIR-only.
Start from Phase 1 best.pt, lower lr, no mixup, fewer epochs."""
from ultralytics import YOLO

model = YOLO("/workspace/v1/runs/teacher_x_1536_all/weights/best.pt")
results = model.train(
    data="/workspace/v1/data3/data/data_dprmir.yaml",
    epochs=8,
    imgsz=1536,
    batch=12,
    device=0,
    project="/workspace/v1/runs",
    name="teacher_x_1536_dprft",
    exist_ok=True,
    workers=8,
    cache=False,
    # Fine-tune augs — softer than Phase 1
    mosaic=1.0,
    mixup=0.0,           # DISABLED (was 0.15)
    copy_paste=0.1,      # reduced (was 0.3)
    degrees=5,           # reduced (was 10)
    translate=0.05,      # reduced (was 0.1)
    scale=0.3,           # reduced (was 0.5)
    fliplr=0.5,
    flipud=0.0,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    close_mosaic=3,      # close mosaic earlier (was 10)
    # Fine-tune lr — 10x lower
    optimizer="AdamW",
    lr0=0.0001,          # was 0.001
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=0,     # no warmup for fine-tune
    cos_lr=True,
    patience=4,          # stop early if plateau
    amp=True,
    save=True, save_period=2, plots=True, val=True,
)
print("TEACHER PHASE 2 DONE")
