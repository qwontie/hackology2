from ultralytics import YOLO
from pathlib import Path



EPOCHS = 100
IMGSZ  = 640
BATCH  = 16  

model1 = YOLO("yolov8s.pt")
model1.train(
    data="bottles.yaml",
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    seed=42,
    fl_gamma=2.0,
    # аугментация
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    fliplr=0.5,  flipud=0.0,
    mosaic=1.0,  mixup=0.0,
    # сохранение
    project="runs", name="model1",
    save=True,
)

model2 = YOLO("yolov8m.pt")
model2.train(
    data="bottles.yaml",
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    seed=123,
    fl_gamma=2.0,
    hsv_h=0.05, hsv_s=0.9, hsv_v=0.6,
    fliplr=0.5, flipud=0.5,
    mosaic=1.0, mixup=0.2,
    project="runs", name="model2",
    save=True,
)

model3 = YOLO("rtdetr-l.pt")
model3.train(
    data="bottles.yaml",
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    seed=777,
    fl_gamma=2.0,
    degrees=15.0, translate=0.2,
    scale=0.6,    shear=5.0,
    mosaic=0.5,
    project="runs", name="model3",
    save=True,
)

print("Done! Weights are saved")
for name in ["model1", "model2", "model3"]:
    p = Path(f"runs/{name}/weights/best.pt")
    print(f"  {p}  {'✓' if p.exists() else '✗ 'NOT FOUND'}")  