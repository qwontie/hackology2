"""Compare Phase 1 vs Phase 2 teacher on the same val set."""
from ultralytics import YOLO

VAL_YAML = "/workspace/v1/data3/data/data.yaml"  # val: val.txt (149 DPR_MIR)

print("=" * 60)
print("Phase 1 teacher (teacher_x_1536_all/best.pt)")
print("=" * 60)
m1 = YOLO("/workspace/v1/runs/teacher_x_1536_all/weights/best.pt")
r1 = m1.val(data=VAL_YAML, imgsz=1536, batch=12, device=0, half=True, verbose=False, plots=False, save_json=False)
print(f"mAP50    = {r1.box.map50:.4f}")
print(f"mAP50-95 = {r1.box.map:.4f}")
print(f"mP       = {r1.box.mp:.4f}")
print(f"mR       = {r1.box.mr:.4f}")

print()
print("=" * 60)
print("Phase 2 teacher (teacher_x_1536_dprft/best.pt)")
print("=" * 60)
m2 = YOLO("/workspace/v1/runs/teacher_x_1536_dprft/weights/best.pt")
r2 = m2.val(data=VAL_YAML, imgsz=1536, batch=12, device=0, half=True, verbose=False, plots=False, save_json=False)
print(f"mAP50    = {r2.box.map50:.4f}")
print(f"mAP50-95 = {r2.box.map:.4f}")
print(f"mP       = {r2.box.mp:.4f}")
print(f"mR       = {r2.box.mr:.4f}")

print()
print("=" * 60)
print(f"DELTA (Phase 2 - Phase 1)")
print(f"  mAP50    = {r2.box.map50 - r1.box.map50:+.4f}")
print(f"  mAP50-95 = {r2.box.map - r1.box.map:+.4f}")
print("=" * 60)
