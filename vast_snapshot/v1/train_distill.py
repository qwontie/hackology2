"""Phase 3: Distill yolo11x teacher -> yolo11m student via CWD.

Teacher: Phase 1 best.pt (mAP50=0.9253)
Student: yolo11m.pt (COCO pretrained)
Loss: CWD (Channel-Wise Distillation) on layers 6/8/13/16/19/22
"""
import sys
from ultralytics import YOLO

SMOKE = "--smoke" in sys.argv

TEACHER_PATH = "/workspace/v1/runs/teacher_x_1536_all/weights/best.pt"
STUDENT_INIT = "/workspace/weights/yolo11m.pt"
DATA = "/workspace/v1/data3/data/data.yaml"
RUN_NAME = "student_m_1536_cwd" + ("_smoke" if SMOKE else "")

print(f"Teacher: {TEACHER_PATH}")
print(f"Student init: {STUDENT_INIT}")
print(f"Run: {RUN_NAME}")
print(f"Smoke test: {SMOKE}")

teacher = YOLO(TEACHER_PATH)
student = YOLO(STUDENT_INIT)

results = student.train(
    data=DATA,
    teacher=teacher.model,
    distillation_loss="cwd",
    epochs=2 if SMOKE else 35,
    imgsz=1536,
    batch=8,                 # conservative; teacher+student+hooks at 1536
    device=0,
    project="/workspace/v1/runs",
    name=RUN_NAME,
    exist_ok=True,
    workers=4,               # README suggests 0 for hooks; trying 4 first
    cache=False,
    # Augmentations — same as Phase 1
    mosaic=1.0,
    mixup=0.15,
    copy_paste=0.3,
    degrees=10,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    flipud=0.0,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    close_mosaic=10,
    # LR
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=3,
    cos_lr=True,
    patience=15,
    amp=True,
    save=True,
    save_period=5,
    plots=True,
    val=True,
)
print("STUDENT DISTILL DONE")
