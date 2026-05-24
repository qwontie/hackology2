"""Enhanced inference: presets + TTA + WBF + multi-model ensemble + degenerate filter.

Drop-in compatible CLI with baseline predict.py, plus new flags:
  --mode safe|balanced|heavy   inference profile for T4 16GB/30min budget
  --models NAME [NAME ...]     ensemble multiple weights via WBF
  --imgsz INT                  override resolution
  --tta-flip                   horizontal flip TTA
  --tta-scales 1280 1536       multi-scale TTA
  --wbf-iou FLOAT              IoU thresh for Weighted Box Fusion
  --max-det INT                max boxes per image (300/400/500)

Model name resolution (--model / --models accepts):
  - alias: 'student' | 'teacher' | 'teacher_dprft' (auto-downloads from GH Release)
  - URL: https://...something.pt
  - local path: _weights/foo.pt

Modes are presets that set the above flags (timings: 481 imgs on H100, T4 ~2-3x slower):
  safe:     single-scale 1280, no TTA, max_det=300
  balanced: single-scale 1536 + hflip, max_det=400
  heavy:    multi-scale [1280,1536,1920] + hflip + WBF, max_det=500

For T4 finals (16GB, 30min), profile on Colab first.

IMPORTANT — ultralytics `augment=True` is NOT just a horizontal flip. It triggers
the built-in TTA (3 scale-pass + flip ≈ 4 forward passes). So `heavy` mode with
external `scales=[1280, 1536, 1920]` and `ult_tta=True` does 3 × 4 = 12 forward
passes PER MODEL. With 2 models = 24 fwd passes per image — confirmed 35 min on
T4. DO NOT use heavy on T4.

Mode pass-count per model (per image):
  safe       1   (single imgsz=1280, no TTA)
  balanced   4   (imgsz=1536 + ult_tta — ultralytics 3-scale + flip)
  semiheavy  2   (external scales [1280, 1536], no ult_tta — predictable)
  heavy      12  (external scales × ult_tta — DON'T use on T4)

ALWAYS filters w<1 or h<1 bboxes (eval validator rejects them — see epoch5 incident).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import torch
from ultralytics import YOLO, RTDETR

# Optional sibling-copy postprocess (codex V3 — targets 1L/700ml/500ml SKU confusions).
# Enable with --sibling-copy or PREDICT_SIBLING_COPY=1. Adds shadow preds at a
# discounted score for known sibling pairs; mAP gain ~+0.004 from 3 weak classes.
try:
    from scripts.sibling_copy import apply_sibling_copy
    HAS_SIBLING = True
except ImportError:
    HAS_SIBLING = False


# ---------- weight registry: short names -> GH Release URLs ----------
WEIGHTS_RELEASE = "https://github.com/qwontie/hackology2/releases/download/v0-weights"
WEIGHT_ALIASES = {
    "student":       f"{WEIGHTS_RELEASE}/student_m_1536_cwd_best.pt",
    "teacher":       f"{WEIGHTS_RELEASE}/teacher_x_1536_all_best.pt",
    "teacher_dprft": f"{WEIGHTS_RELEASE}/teacher_x_1536_dprft_best.pt",
    # uploaded after training finishes (auto-download once GH Release has them):
    "yolov8l":       f"{WEIGHTS_RELEASE}/yolov8l_train1_best.pt",
    "yolo11l":       f"{WEIGHTS_RELEASE}/yolo11l_student_best.pt",
    "rtdetr":        f"{WEIGHTS_RELEASE}/rtdetr_l_best.pt",
    # +0.0074 diversity hero (4-model champion 0.8116 -> public LB):
    "yolo11x_v2":    f"{WEIGHTS_RELEASE}/yolo11x_v2_best.pt",
    # +0.0056 diversity hero (5-model champion 0.8172 -> public LB):
    "yolov8x_cb":    f"{WEIGHTS_RELEASE}/yolov8x_cb_best.pt",
}
WEIGHTS_CACHE_DIR = Path(os.environ.get("WEIGHTS_CACHE_DIR", "_weights"))

# Finals invocation has no flags — `uv run predict --input ... --annotations ... --output ...`.
# These env vars (or hardcoded fallbacks) decide what runs in production.
# Default mode is `balanced` (1536+flip per model), not `heavy`. T4 smoke test
# at 2026-05-24 02:30 measured heavy 2-model = 35 min, OVER the 30-min finals
# budget. balanced 2-model = ~12 min, safe headroom. Override with PREDICT_MODE=heavy
# only if you've actually re-profiled on the target box.
DEFAULT_MODE = os.environ.get("PREDICT_MODE", "semiheavy")
# 5-model public-LB champion @ 0.8172. Each addition was verified +δ:
#   student+teacher baseline, +yolo11l (+0.017), +yolo11x_v2 (+0.0074), +yolov8x_cb (+0.0056).
DEFAULT_MODELS = os.environ.get("PREDICT_MODELS", "student teacher yolo11l yolo11x_v2 yolov8x_cb").split()
DEFAULT_MODEL_WEIGHTS_ENV = os.environ.get("PREDICT_MODEL_WEIGHTS", "1.0 1.5 0.7 0.5 0.5")


def is_rtdetr_weight(name_or_path: str) -> bool:
    """Detect rtdetr by alias or filename — they need the RTDETR class, not YOLO."""
    return "rtdetr" in name_or_path.lower()


def load_model(path: str):
    """Pick the right ultralytics class so loss/inference paths are correct."""
    if is_rtdetr_weight(path):
        return RTDETR(path)
    return YOLO(path)


def resolve_weight(name_or_path: str) -> str:
    """Resolve a weight name to a local path. Downloads from GH Release if needed.

    Accepts: alias (student/teacher/teacher_dprft), URL, or local path.
    Returns: local file path.
    """
    if name_or_path in WEIGHT_ALIASES:
        url = WEIGHT_ALIASES[name_or_path]
        local = WEIGHTS_CACHE_DIR / Path(url).name
    elif name_or_path.startswith(("http://", "https://")):
        url = name_or_path
        local = WEIGHTS_CACHE_DIR / Path(url).name
    else:
        # Local path
        local = Path(name_or_path)
        if not local.exists():
            print(f"ERROR: weight file not found: {local}", file=sys.stderr)
            sys.exit(1)
        return str(local)

    if local.exists():
        return str(local)

    WEIGHTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {local}", file=sys.stderr)
    t0 = time.time()
    urllib.request.urlretrieve(url, str(local))
    size_mb = local.stat().st_size / 1024**2
    print(f"  done ({size_mb:.1f}MB in {time.time()-t0:.1f}s)", file=sys.stderr)
    return str(local)

try:
    from ensemble_boxes import weighted_boxes_fusion
    HAS_WBF = True
except ImportError:
    HAS_WBF = False


# ---------- mode presets ----------
MODES = {
    # `ult_tta` controls ultralytics built-in TTA (augment=True ≈ 3-scale + flip).
    # `scales` is our explicit external loop. Total passes/model = len(scales or [imgsz]) × (4 if ult_tta else 1).
    "safe":      {"imgsz": 1280, "ult_tta": False, "scales": None,               "max_det": 300},  # 1
    "balanced":  {"imgsz": 1536, "ult_tta": True,  "scales": None,               "max_det": 400},  # 4
    "semiheavy": {"imgsz": 1536, "ult_tta": False, "scales": [1280, 1536],       "max_det": 500},  # 2 — predictable, T4-safe
    "heavy":     {"imgsz": 1536, "ult_tta": True,  "scales": [1280, 1536, 1920], "max_det": 500},  # 12 — H100 only
}


# ---------- helpers ----------
def load_taxonomy(path: Path) -> dict[int, int]:
    """Map YOLO class index (0..N-1) -> hackathon category_id from taxonomy.json."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    cats = sorted(data["categories"], key=lambda c: c["id"])
    return {i: c["id"] for i, c in enumerate(cats)}


def load_image_id_map(annotations_path: Path) -> dict[str, int]:
    """Map filename -> image_id from test_images.json (COCO format)."""
    if not annotations_path.exists():
        print(f"WARNING: {annotations_path} not found", file=sys.stderr)
        return {}
    data = json.loads(annotations_path.read_text(encoding="utf-8"))
    return {img["file_name"]: img["id"] for img in data["images"]}


def boxes_to_normalized(boxes_xyxy, img_w: int, img_h: int):
    """xyxy in pixels -> normalized [0,1] format expected by ensemble_boxes."""
    out = []
    for x1, y1, x2, y2 in boxes_xyxy:
        out.append([
            max(0.0, min(1.0, x1 / img_w)),
            max(0.0, min(1.0, y1 / img_h)),
            max(0.0, min(1.0, x2 / img_w)),
            max(0.0, min(1.0, y2 / img_h)),
        ])
    return out


def predict_one_pass(model: YOLO, img_path: str, imgsz: int, conf: float,
                     max_det: int, ult_tta: bool, nms_iou: float = 0.7,
                     device: int = 0) -> tuple[list, list, list]:
    """Single inference pass. If ult_tta=True, uses ultralytics built-in TTA
    (augment=True ≈ 3 internal scales + flip → ~4× forward passes). DO NOT
    confuse this with horizontal flip alone.

    nms_iou: ultralytics NMS IoU threshold for per-model dedup BEFORE WBF.
    Default 0.7 = ultralytics default. Higher (0.85-0.90) keeps more candidates
    for WBF — codex says this matters more than wbf_iou tuning alone.

    Returns: (xyxy_list, cls_list, conf_list) in original-image pixel coords.
    """
    res = model.predict(source=img_path, imgsz=imgsz, conf=conf, max_det=max_det,
                        iou=nms_iou, device=device, verbose=False, half=True,
                        augment=ult_tta)
    boxes = res[0].boxes
    xyxy = boxes.xyxy.cpu().tolist() if boxes is not None else []
    cls = boxes.cls.cpu().tolist() if boxes is not None else []
    cf = boxes.conf.cpu().tolist() if boxes is not None else []
    del res
    return xyxy, cls, cf


def wbf_merge(all_xyxy: list[list], all_cls: list[list], all_conf: list[list],
              img_w: int, img_h: int, iou_thr: float = 0.55,
              weights: list[float] | None = None) -> tuple[list, list, list]:
    """Merge predictions from multiple passes/models via WBF. Returns pixel-space boxes."""
    if not HAS_WBF:
        # Fallback: concatenate (NMS already done per-pass by YOLO)
        return ([b for xs in all_xyxy for b in xs],
                [c for cs in all_cls for c in cs],
                [c for cs in all_conf for c in cs])
    if not any(all_xyxy):
        return [], [], []

    norm = [boxes_to_normalized(xs, img_w, img_h) for xs in all_xyxy]
    boxes, scores, labels = weighted_boxes_fusion(
        norm, all_conf, all_cls,
        weights=weights, iou_thr=iou_thr, skip_box_thr=0.001,
    )
    # Back to pixels
    out_xyxy = [[b[0]*img_w, b[1]*img_h, b[2]*img_w, b[3]*img_h] for b in boxes]
    return out_xyxy, list(labels), list(scores)


def predict_all(
    models: list,
    img_paths: list[Path],
    fname_to_id: dict[str, int],
    idx_to_cat: dict[int, int],
    imgsz: int,
    ult_tta: bool,
    scales: list[int] | None,
    max_det: int,
    conf: float,
    wbf_iou: float,
    nms_iou: float = 0.7,
    model_weights: list[float] | None = None,
    score_power: float = 1.0,
) -> list[dict]:
    """Main inference loop with memory cleanup every 50 imgs.

    model_weights: per-MODEL weights, broadcast across scales when WBF-fusing.
    None == all weights = 1.
    score_power: gamma applied to each pred's score BEFORE WBF. <1.0 boosts
    low-conf preds asymmetrically (e.g. 0.6 maps 0.001->0.016, 0.5->0.66,
    0.9->0.94). Rank-changing in WBF context — can rescue low-conf TPs that
    skip_box_thr would otherwise drop. 1.0 = no-op.
    """
    preds = []
    t0 = time.time()
    # Get image sizes lazily for WBF normalization
    from PIL import Image  # local import
    sizes_cache: dict[str, tuple[int, int]] = {}

    for i, img_path in enumerate(img_paths):
        image_id = fname_to_id.get(img_path.name)
        if image_id is None:
            continue

        # All passes for this image
        all_xyxy, all_cls, all_conf, pass_weights = [], [], [], []
        sz_list = scales if scales else [imgsz]
        for sz in sz_list:
            for m_idx, model in enumerate(models):
                xyxy, cls, cf = predict_one_pass(model, str(img_path), sz, conf, max_det, ult_tta, nms_iou=nms_iou)
                if xyxy:
                    if score_power != 1.0:
                        cf = [c ** score_power for c in cf]
                    all_xyxy.append(xyxy)
                    all_cls.append(cls)
                    all_conf.append(cf)
                    pass_weights.append(model_weights[m_idx] if model_weights else 1.0)

        # Merge
        if len(all_xyxy) == 0:
            continue
        if len(all_xyxy) == 1 and not ult_tta:
            xyxy, cls, conf_list = all_xyxy[0], all_cls[0], all_conf[0]
        else:
            if img_path.name not in sizes_cache:
                with Image.open(img_path) as im:
                    sizes_cache[img_path.name] = im.size  # (W, H)
            w, h = sizes_cache[img_path.name]
            xyxy, cls, conf_list = wbf_merge(
                all_xyxy, all_cls, all_conf, w, h, wbf_iou,
                weights=pass_weights,
            )

        # Emit COCO entries with degenerate-bbox protection
        for k in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[k]
            bw = x2 - x1
            bh = y2 - y1
            if bw < 1.0 or bh < 1.0:
                continue  # protect against eval validator rejection
            cat_id = idx_to_cat.get(int(cls[k]))
            if cat_id is None:
                continue
            # WBF can sum weighted scores from overlapping models -> >1.0.
            # Eval validator requires score in (0, 1], clip the upper end.
            score = min(max(float(conf_list[k]), 1e-6), 1.0)
            preds.append({
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)],
                "score": round(score, 4),
            })

        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(img_paths) - i - 1) / rate
            mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            print(f"  [{i+1}/{len(img_paths)}] {rate:.2f} img/s eta={eta:.0f}s preds={len(preds)} gpu={mem:.1f}GB",
                  file=sys.stderr, flush=True)

    return preds


# ---------- CLI ----------
def main() -> None:
    p = argparse.ArgumentParser(description="Enhanced YOLO inference with TTA + WBF")
    p.add_argument("--input", type=Path, required=True, help="Directory with input images")
    p.add_argument("--output", type=Path, default=Path("predictions.json"))
    p.add_argument("--model", type=str, help="Single model path (legacy compat)")
    p.add_argument("--models", nargs="+", help="Multiple model paths for ensemble WBF")
    p.add_argument("--confidence", type=float, default=0.001,
                   help="Minimum confidence (default: 0.001, low for mAP)")
    p.add_argument("--taxonomy", type=Path, default=Path("taxonomy.json"))
    p.add_argument("--annotations", type=Path, default=Path("test_images.json"))
    p.add_argument("--mode", choices=list(MODES.keys()), default=DEFAULT_MODE,
                   help=f"Preset profile (default from PREDICT_MODE env or '{DEFAULT_MODE}')")
    p.add_argument("--imgsz", type=int)
    p.add_argument("--tta-flip", action="store_true")
    p.add_argument("--tta-scales", type=int, nargs="+")
    p.add_argument("--wbf-iou", type=float, default=0.55)
    p.add_argument("--nms-iou", type=float,
                   default=float(os.environ.get("PREDICT_NMS_IOU", "0.7")),
                   help="Ultralytics NMS IoU for per-model dedup BEFORE WBF. "
                        "Codex: 0.85 serious candidate, 0.90 worth probe. "
                        "Higher = more candidates fed to WBF.")
    p.add_argument("--max-det", type=int)
    p.add_argument("--model-weights", type=float, nargs="+",
                   help="Per-model weights for WBF (default: all 1.0, or set PREDICT_MODEL_WEIGHTS env)")
    p.add_argument("--sibling-copy", action="store_true",
                   default=os.environ.get("PREDICT_SIBLING_COPY") == "1",
                   help="Apply sibling-class shadow postprocess (codex V3 targeted 1L/700ml fix). "
                        "Default from PREDICT_SIBLING_COPY env.")
    p.add_argument("--score-power", type=float,
                   default=float(os.environ.get("PREDICT_SCORE_POWER", "1.0")),
                   help="Gamma applied to per-model scores BEFORE WBF (default 1.0 = no-op). "
                        "0.5-0.7 boosts low-conf preds asymmetrically and reranks WBF fusion. "
                        "Env: PREDICT_SCORE_POWER.")
    args = p.parse_args()

    # Apply mode preset, allow override by explicit flags
    preset = MODES[args.mode]
    imgsz = args.imgsz if args.imgsz else preset["imgsz"]
    ult_tta = args.tta_flip if args.tta_flip else preset["ult_tta"]
    scales = args.tta_scales if args.tta_scales else preset["scales"]
    max_det = args.max_det if args.max_det else preset["max_det"]

    # Models — fall back to env-configured defaults so finals (no flags) just works
    model_paths = args.models or ([args.model] if args.model else None) or DEFAULT_MODELS
    if not model_paths:
        print("ERROR: no models specified and PREDICT_MODELS env is empty", file=sys.stderr)
        sys.exit(1)

    # Per-model WBF weights (CLI > env > all 1.0)
    model_weights = args.model_weights
    if model_weights is None and DEFAULT_MODEL_WEIGHTS_ENV:
        try:
            model_weights = [float(x) for x in DEFAULT_MODEL_WEIGHTS_ENV.split()]
        except ValueError:
            print(f"WARNING: bad PREDICT_MODEL_WEIGHTS='{DEFAULT_MODEL_WEIGHTS_ENV}', using unit weights",
                  file=sys.stderr)
            model_weights = None
    if model_weights and len(model_weights) != len(model_paths):
        print(f"ERROR: --model-weights has {len(model_weights)} entries but {len(model_paths)} models",
              file=sys.stderr)
        sys.exit(1)

    print(f"Mode: {args.mode} | imgsz={imgsz} ult_tta={ult_tta} scales={scales} max_det={max_det}",
          file=sys.stderr)
    print(f"Models: {model_paths} weights={model_weights or 'unit'} score_power={args.score_power}",
          file=sys.stderr)
    if (ult_tta or scales or len(model_paths) > 1) and not HAS_WBF:
        print("WARNING: ensemble_boxes not installed, falling back to concat (may degrade mAP)",
              file=sys.stderr)

    resolved_paths = [resolve_weight(pth) for pth in model_paths]
    print(f"Resolved: {resolved_paths}", file=sys.stderr)
    models = [load_model(pth) for pth in resolved_paths]

    # Image discovery
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    fname_to_id = load_image_id_map(args.annotations)
    idx_to_cat = load_taxonomy(args.taxonomy)
    img_paths = sorted(p for p in args.input.iterdir()
                       if p.suffix.lower() in exts and p.name in fname_to_id)
    if not img_paths:
        print(f"ERROR: no matching images in {args.input}", file=sys.stderr); sys.exit(1)
    print(f"Images: {len(img_paths)}", file=sys.stderr)

    # Run
    t0 = time.time()
    preds = predict_all(models, img_paths, fname_to_id, idx_to_cat,
                        imgsz=imgsz, ult_tta=ult_tta, scales=scales, max_det=max_det,
                        conf=args.confidence, wbf_iou=args.wbf_iou, nms_iou=args.nms_iou,
                        model_weights=model_weights, score_power=args.score_power)

    if args.sibling_copy:
        if not HAS_SIBLING:
            print("WARNING: --sibling-copy requested but scripts/sibling_copy.py not importable",
                  file=sys.stderr)
        else:
            before = len(preds)
            preds, sib_stats = apply_sibling_copy(preds)
            print(f"sibling-copy: {before:,} -> {len(preds):,} (+{sib_stats['added_total']:,}) "
                  f"by_pair={sib_stats['by_pair']}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(preds))
    print(f"\nDONE {time.time()-t0:.0f}s | {len(preds):,} preds | {args.output} "
          f"({args.output.stat().st_size/1024/1024:.2f}MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
