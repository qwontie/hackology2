"""Phase 1: teacher yolo11x @ 1536, all 3754 images, 40 epochs."""
from ultralytics import YOLO

model = YOLO("/workspace/weights/yolo11x.pt")

results = model.train(
    data="/workspace/v1/data3/data/data.yaml",
    epochs=40,
    imgsz=1536,
    batch=12,
    device=0,
    project="/workspace/v1/runs",
    name="teacher_x_1536_all",
    exist_ok=True,
    workers=8,
    cache=False,

    # augmentations — strong for crowded shelf
    mosaic=1.0,
    mixup=0.15,
    copy_paste=0.3,
    degrees=10,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    flipud=0.0,         # bottles upright
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    close_mosaic=10,    # off last 10 epochs

    # optimization
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=3,
    cos_lr=True,
    patience=25,
    amp=True,

    # save
    save=True,
    save_period=5,
    plots=True,
    val=True,
)
print("TEACHER PHASE 1 DONE")
