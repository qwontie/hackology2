"""COCO -> YOLO conversion + DPR_MIR-based val split."""
import json, random, yaml
from pathlib import Path
from collections import Counter

BASE = Path("/workspace/v1/data3/data")
TRAIN_DIR = BASE / "train"
IMG_DIR = TRAIN_DIR / "images"
LBL_DIR = TRAIN_DIR / "labels"
LBL_DIR.mkdir(exist_ok=True)

TAXONOMY = Path("/workspace/v1/taxonomy.json")
# Fallback: try same dir as annotations
if not TAXONOMY.exists():
    TAXONOMY = TRAIN_DIR / "taxonomy.json"

DATA_YAML = BASE / "data.yaml"
DATA_YAML_DPRMIR = BASE / "data_dprmir.yaml"
TRAIN_TXT = BASE / "train.txt"
VAL_TXT = BASE / "val.txt"
DPRMIR_TRAIN_TXT = BASE / "dprmir_train.txt"
DPRMIR_VAL_TXT = BASE / "dprmir_val.txt"

VAL_FRAC_DPR = 150  # absolute number of DPR_MIR imgs held out for val

print(f"Loading annotations...")
with open(TRAIN_DIR / "annotations.json") as f:
    ann = json.load(f)
with open(TAXONOMY) as f:
    tax = json.load(f)

cats = sorted(tax["categories"], key=lambda c: c["id"])
cat_id_to_idx = {c["id"]: i for i, c in enumerate(cats)}
names = {i: c["name"] for i, c in enumerate(cats)}
print(f"  {len(cats)} categories")

img_by_id = {im["id"]: im for im in ann["images"]}
anns_by_img = {}
for a in ann["annotations"]:
    anns_by_img.setdefault(a["image_id"], []).append(a)

# Split: hold out 150 DPR_MIR images as val
random.seed(42)
dpr_mir_imgs = [im for im in ann["images"]
                if im.get("source_dataset") == "DPR_MIR_3963_QUALITY_EVALUATION_09072025"]
random.shuffle(dpr_mir_imgs)
val_ids = {im["id"] for im in dpr_mir_imgs[:VAL_FRAC_DPR]}
dpr_train_ids = {im["id"] for im in dpr_mir_imgs[VAL_FRAC_DPR:]}

print(f"  DPR_MIR total: {len(dpr_mir_imgs)}, val held out: {len(val_ids)}, dpr_train: {len(dpr_train_ids)}")

# Write YOLO labels
written = 0
missing_imgs = 0
for im in ann["images"]:
    img_path = IMG_DIR / im["file_name"]
    if not img_path.exists():
        missing_imgs += 1
        continue
    lines = []
    for a in anns_by_img.get(im["id"], []):
        idx = cat_id_to_idx.get(a["category_id"])
        if idx is None:
            continue
        x, y, w, h = a["bbox"]
        # convert to YOLO: normalized cx, cy, w, h
        cx = (x + w / 2) / im["width"]
        cy = (y + h / 2) / im["height"]
        nw = w / im["width"]
        nh = h / im["height"]
        # clamp
        cx = min(max(cx, 0), 1)
        cy = min(max(cy, 0), 1)
        nw = min(max(nw, 0), 1)
        nh = min(max(nh, 0), 1)
        lines.append(f"{idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    lbl_path = LBL_DIR / (Path(im["file_name"]).stem + ".txt")
    lbl_path.write_text("\n".join(lines))
    written += 1

print(f"  wrote {written} label files, {missing_imgs} images missing on disk")

# Write train/val file lists (absolute paths)
all_train_paths = []
all_val_paths = []
dpr_train_paths = []
dpr_val_paths = []
for im in ann["images"]:
    p = str(IMG_DIR / im["file_name"])
    if not (IMG_DIR / im["file_name"]).exists():
        continue
    if im["id"] in val_ids:
        all_val_paths.append(p)
        dpr_val_paths.append(p)
    else:
        all_train_paths.append(p)
        if im["id"] in dpr_train_ids:
            dpr_train_paths.append(p)

TRAIN_TXT.write_text("\n".join(all_train_paths))
VAL_TXT.write_text("\n".join(all_val_paths))
DPRMIR_TRAIN_TXT.write_text("\n".join(dpr_train_paths))
DPRMIR_VAL_TXT.write_text("\n".join(dpr_val_paths))

print(f"  Phase 1 (all): train={len(all_train_paths)}, val={len(all_val_paths)}")
print(f"  Phase 2 (DPR_MIR only): train={len(dpr_train_paths)}, val={len(dpr_val_paths)}")

# Write data.yaml files
with open(DATA_YAML, "w") as f:
    yaml.safe_dump({
        "path": str(BASE),
        "train": str(TRAIN_TXT),
        "val": str(VAL_TXT),
        "names": names,
    }, f, sort_keys=False)

with open(DATA_YAML_DPRMIR, "w") as f:
    yaml.safe_dump({
        "path": str(BASE),
        "train": str(DPRMIR_TRAIN_TXT),
        "val": str(DPRMIR_VAL_TXT),
        "names": names,
    }, f, sort_keys=False)

print(f"  wrote {DATA_YAML} and {DATA_YAML_DPRMIR}")
print("DONE")
