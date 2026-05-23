"""Model soup via YOLO loader — handles both full and stripped checkpoints."""
import shutil
from pathlib import Path
import torch
from ultralytics import YOLO

W = Path("/workspace/v1/runs/teacher_x_1536_all/weights")
OUT = Path("/workspace/v1/runs/soup")
OUT.mkdir(exist_ok=True)
VAL_YAML = "/workspace/v1/data3/data/data.yaml"


def avg_state_dicts(sds):
    """Average a list of state_dicts (preserve dtype, skip int buffers)."""
    out = {}
    for k in sds[0]:
        v0 = sds[0][k]
        if v0.dtype.is_floating_point:
            stacked = torch.stack([sd[k].float() for sd in sds])
            out[k] = stacked.mean(dim=0).to(v0.dtype)
        else:
            out[k] = v0
    return out


def soup(paths, out_path, template_path):
    """Average weights from `paths` and save as `out_path` using `template_path` as ckpt skeleton."""
    print(f"  Loading {len(paths)} checkpoints: {[p.name for p in paths]}", flush=True)
    sds = []
    for p in paths:
        m = YOLO(str(p))
        sds.append({k: v.detach().clone() for k, v in m.model.state_dict().items()})
    avg = avg_state_dicts(sds)

    # Build target model from template, load averaged weights, save via YOLO
    shutil.copy(template_path, out_path)
    target = YOLO(str(out_path))
    target.model.load_state_dict(avg, strict=True)
    target.save(str(out_path))
    print(f"  Saved {out_path.name} ({out_path.stat().st_size / 1024**2:.1f}MB)", flush=True)


def validate(path, label):
    print(f"\n>>> {label}", flush=True)
    m = YOLO(str(path))
    r = m.val(data=VAL_YAML, imgsz=1536, batch=12, device=0, half=True,
              verbose=False, plots=False, save_json=False)
    print(f"  mAP50    = {r.box.map50:.4f}", flush=True)
    print(f"  mAP50-95 = {r.box.map:.4f}", flush=True)
    print(f"  mP/mR    = {r.box.mp:.4f} / {r.box.mr:.4f}", flush=True)
    return r.box.map50, r.box.map


res = {}
res["ep38 (best.pt baseline)"] = validate(W / "best.pt", "ep38 baseline")

combos = [
    ("soup_30_35_38_40.pt", [W / "epoch30.pt", W / "epoch35.pt", W / "best.pt", W / "last.pt"]),
    ("soup_35_38_40.pt",    [W / "epoch35.pt", W / "best.pt", W / "last.pt"]),
    ("soup_30_35_38.pt",    [W / "epoch30.pt", W / "epoch35.pt", W / "best.pt"]),
    ("soup_38_40.pt",       [W / "best.pt", W / "last.pt"]),
]

template = W / "best.pt"  # use stripped best.pt as the skeleton
for name, paths in combos:
    out = OUT / name
    print(f"\n=== {name} ===", flush=True)
    soup(paths, out, template)
    res[name] = validate(out, name)

print("\n" + "=" * 72)
print(f"{'config':<37} {'mAP50':>10} {'mAP50-95':>10} {'Δ50':>10}")
print("-" * 72)
base_map50 = res["ep38 (best.pt baseline)"][0]
for k, (m50, m9595) in res.items():
    delta = m50 - base_map50
    flag = " ★" if delta > 0.001 else ""
    print(f"{k:<37} {m50:>10.4f} {m9595:>10.4f} {delta:>+10.4f}{flag}")
print("=" * 72)
