"""After Phase 3 distill: validate student vs teacher head-to-head on val + DPR_MIR-only."""
from ultralytics import YOLO

DATA_FULL  = "/workspace/v1/data3/data/data.yaml"
DATA_DPR   = "/workspace/v1/data3/data/data_dprmir.yaml"
TEACHER    = "/workspace/v1/runs/teacher_x_1536_all/weights/best.pt"
STUDENT    = "/workspace/v1/runs/student_m_1536_cwd/weights/best.pt"

def run(label, model_path, data_path, imgsz=1536):
    m = YOLO(model_path)
    r = m.val(data=data_path, imgsz=imgsz, batch=12, device=0, half=True,
              verbose=False, plots=False, save_json=False)
    print(f"{label:<40} mAP50={r.box.map50:.4f}  mAP50-95={r.box.map:.4f}  "
          f"mP={r.box.mp:.4f}  mR={r.box.mr:.4f}", flush=True)
    return r.box.map50

print("=" * 80)
print(f"{'config':<40} {'metrics':<40}")
print("-" * 80)
t_full = run("teacher_x @1536 (full val)",  TEACHER, DATA_FULL)
s_full = run("student_m @1536 (full val)",  STUDENT, DATA_FULL)
print(f"{'  delta student-teacher':<40} {s_full - t_full:+.4f}")
print("-" * 80)
t_d   = run("teacher_x @1536 (DPR_MIR val)", TEACHER, DATA_DPR)
s_d   = run("student_m @1536 (DPR_MIR val)", STUDENT, DATA_DPR)
print(f"{'  delta student-teacher (DPR)':<40} {s_d - t_d:+.4f}")
print("=" * 80)
