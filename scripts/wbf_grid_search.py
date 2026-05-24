"""B6 WBF grid search (codex sweep ranges).

Loads cached per-model JSONL predictions (from cache_val_preds.py), fuses
via WBF with different param combos, evaluates AP@0.5 against val COCO GT.

Sweeps per codex recommendation:
  iou_thr:       0.50, 0.55, 0.60, 0.65
  skip_box_thr:  0.001, 0.003, 0.005
  model weights: user-supplied grid

Eval via pycocotools COCOeval (IoU=0.5 only, fast).

Usage:
    python scripts/wbf_grid_search.py \\
        --cached /tmp/val_preds_cache \\
        --val-anno data/val/annotations.json \\
        --models student_m_1536_cwd_best teacher_x_1536_all_best yolo11l_student_best \\
        --weight-grid '{"student":[0.8,1.0,1.2],"teacher":[1.2,1.5,1.8],"y11l":[0.5,0.7,1.0]}' \\
        --topk 30 \\
        --out /tmp/wbf_grid_results.json

Cached JSONL filename format: <model_stem>__imgsz<N>.jsonl
We pick the imgsz that has the most preds (proxy for permissive NMS).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections import defaultdict
from pathlib import Path


def load_cached_preds(cache_dir: Path, model_stems: list[str]) -> dict[str, list[dict]]:
    """Return {model_stem: [{image_id, bbox(xywh), cls_idx (0-based), score}]}."""
    out: dict[str, list[dict]] = {}
    for stem in model_stems:
        candidates = sorted(cache_dir.glob(f"{stem}__imgsz*_iou*.jsonl"))
        if not candidates:
            candidates = sorted(cache_dir.glob(f"{stem}__imgsz*.jsonl"))  # legacy
        if not candidates:
            raise FileNotFoundError(f"No cached preds for model stem '{stem}' in {cache_dir}")
        # Pick the file with the most lines (most permissive NMS / scale)
        pick = max(candidates, key=lambda p: p.stat().st_size)
        rows = [json.loads(l) for l in pick.read_text().splitlines() if l.strip()]
        out[stem] = rows
        print(f"  [{stem}] {pick.name}: {len(rows):,} preds")
    return out


def build_image_dims(anno: dict) -> dict[int, tuple[int, int]]:
    return {im["id"]: (im["width"], im["height"]) for im in anno["images"]}


def group_by_image(preds: list[dict]) -> dict[int, dict]:
    """Return {image_id: {"boxes_norm": [...], "scores": [...], "labels": [...]}}.
    Caller must normalize first via image dims."""
    g: dict[int, dict] = defaultdict(lambda: {"boxes_norm": [], "scores": [], "labels": []})
    for p in preds:
        g[p["image_id"]]["_raw"] = g[p["image_id"]].get("_raw", [])
        g[p["image_id"]]["_raw"].append(p)
    return g


def normalize_box_xywh(bbox: list[float], w: int, h: int) -> list[float]:
    """xywh pixels -> xyxy normalized [0,1]."""
    x, y, bw, bh = bbox
    x1, y1, x2, y2 = x, y, x + bw, y + bh
    return [
        max(0.0, min(1.0, x1 / w)),
        max(0.0, min(1.0, y1 / h)),
        max(0.0, min(1.0, x2 / w)),
        max(0.0, min(1.0, y2 / h)),
    ]


def denormalize_xyxy(boxes_norm: list[list[float]], w: int, h: int) -> list[list[float]]:
    return [[b[0] * w, b[1] * h, b[2] * w, b[3] * h] for b in boxes_norm]


def fuse_one_image(per_model_preds: list[list[dict]], dims: tuple[int, int],
                   weights: list[float], iou_thr: float, skip_box_thr: float):
    """Returns list of COCO-style dicts {image_id?, bbox(xywh), category_id, score}.
    Caller fills image_id and converts cls_idx -> category_id."""
    from ensemble_boxes import weighted_boxes_fusion

    w, h = dims
    boxes_list, scores_list, labels_list = [], [], []
    for preds in per_model_preds:
        if not preds:
            boxes_list.append([])
            scores_list.append([])
            labels_list.append([])
            continue
        boxes_list.append([normalize_box_xywh(p["bbox"], w, h) for p in preds])
        scores_list.append([float(p["score"]) for p in preds])
        labels_list.append([int(p["cls_idx"]) for p in preds])

    if all(len(b) == 0 for b in boxes_list):
        return [], [], []

    fb, fs, fl = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        weights=weights, iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
        conf_type="avg",
    )
    # Denormalize and convert to xywh
    xyxy = denormalize_xyxy(fb.tolist(), w, h)
    xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in xyxy]
    return xywh, fs.tolist(), fl.tolist()


def fuse_all(per_model_cached: dict[str, list[dict]], dims_by_img: dict[int, tuple[int, int]],
             model_stems: list[str], weights: list[float],
             iou_thr: float, skip_box_thr: float,
             cat_idx_to_id: dict[int, int]) -> list[dict]:
    """Fuse all images, return COCO-format predictions list."""
    # Group preds per model per image
    grouped: dict[str, dict[int, list[dict]]] = {}
    for stem in model_stems:
        d: dict[int, list[dict]] = defaultdict(list)
        for p in per_model_cached[stem]:
            d[p["image_id"]].append(p)
        grouped[stem] = d

    all_image_ids = sorted(dims_by_img.keys())
    coco_preds: list[dict] = []
    for img_id in all_image_ids:
        dims = dims_by_img[img_id]
        per_model = [grouped[stem].get(img_id, []) for stem in model_stems]
        if all(len(pm) == 0 for pm in per_model):
            continue
        xywh, scores, labels = fuse_one_image(per_model, dims, weights, iou_thr, skip_box_thr)
        for box, sc, lbl in zip(xywh, scores, labels):
            coco_preds.append({
                "image_id": img_id,
                "category_id": cat_idx_to_id[int(lbl)],
                "bbox": [round(b, 2) for b in box],
                "score": round(float(sc), 5),
            })
    return coco_preds


def eval_coco_ap50(gt_anno_path: Path, predictions: list[dict]) -> float:
    """Return AP@IoU=0.5 (single threshold) using pycocotools."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print("ERROR: pycocotools not installed. pip install pycocotools", file=sys.stderr)
        sys.exit(2)

    coco_gt = COCO(str(gt_anno_path))
    if not predictions:
        return 0.0
    coco_dt = coco_gt.loadRes(predictions)  # type: ignore[arg-type]
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.params.iouThrs = [0.5]   # single threshold = mAP@0.5
    ev.params.areaRng = [[0, 1e10]]
    ev.params.areaRngLbl = ["all"]
    ev.params.maxDets = [600]
    ev.evaluate()
    ev.accumulate()
    # ev.eval['precision'] shape: (T, R, K, A, M) = (1, 101, K, 1, 1)
    precision = ev.eval["precision"]
    # Mean over recall thresholds & classes, ignore -1 (no GT for class)
    precision = precision[0, :, :, 0, 0]   # (R, K)
    valid = precision > -0.5
    if valid.sum() == 0:
        return 0.0
    return float(precision[valid].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached", type=Path, required=True,
                    help="Dir with cached JSONL preds (from cache_val_preds.py)")
    ap.add_argument("--val-anno", type=Path, required=True,
                    help="COCO val annotations.json")
    ap.add_argument("--models", nargs="+", required=True,
                    help="Model stems IN ORDER (filenames sans __imgszNNN.jsonl)")
    ap.add_argument("--short-names", nargs="+", default=None,
                    help="Short labels for weight-grid keys (same order as --models). "
                         "Default: same as --models")
    ap.add_argument("--weight-grid", required=True,
                    help='JSON: {short_name: [w1,w2,...]} — Cartesian product across models')
    ap.add_argument("--iou-thr-grid", nargs="+", type=float, default=[0.50, 0.55, 0.60, 0.65])
    ap.add_argument("--skip-box-grid", nargs="+", type=float, default=[0.001, 0.003, 0.005])
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stage", choices=["weights_only", "full"], default="full",
                    help="weights_only = fix iou=0.55 skip=0.001, sweep weights only (Stage 1). "
                         "full = full Cartesian (slow)")
    args = ap.parse_args()

    short_names = args.short_names if args.short_names else args.models
    if len(short_names) != len(args.models):
        print("--short-names must match --models in length", file=sys.stderr)
        sys.exit(2)

    # Load cached preds
    print(f"[load] cached preds from {args.cached}")
    cached = load_cached_preds(args.cached, args.models)
    n_total = sum(len(v) for v in cached.values())
    print(f"[load] total {n_total:,} preds across {len(args.models)} models")

    # Load GT
    print(f"[load] val GT from {args.val_anno}")
    gt_anno = json.loads(args.val_anno.read_text())
    dims_by_img = build_image_dims(gt_anno)
    # Map YOLO cls_idx (0-based) -> COCO category_id (1-based by id order)
    cats_sorted = sorted(gt_anno["categories"], key=lambda c: c["id"])
    cat_idx_to_id = {i: c["id"] for i, c in enumerate(cats_sorted)}
    print(f"[load] {len(dims_by_img)} val images, {len(cat_idx_to_id)} categories")

    # Build weight combos
    weight_grid = json.loads(args.weight_grid)
    keys = list(short_names)
    for k in keys:
        if k not in weight_grid:
            print(f"weight-grid missing key '{k}'", file=sys.stderr)
            sys.exit(2)
    weight_combos = list(itertools.product(*[weight_grid[k] for k in keys]))

    if args.stage == "weights_only":
        iou_grid = [0.55]
        skip_grid = [0.001]
    else:
        iou_grid = args.iou_thr_grid
        skip_grid = args.skip_box_grid

    n_configs = len(weight_combos) * len(iou_grid) * len(skip_grid)
    print(f"[grid] {len(weight_combos)} weight combos × {len(iou_grid)} iou × {len(skip_grid)} skip = {n_configs} configs")

    results = []
    t_global = time.time()
    for weights in weight_combos:
        for iou_thr in iou_grid:
            for skip in skip_grid:
                t0 = time.time()
                preds = fuse_all(cached, dims_by_img, args.models, list(weights),
                                 iou_thr, skip, cat_idx_to_id)
                ap50 = eval_coco_ap50(args.val_anno, preds)
                dt = time.time() - t0
                results.append({
                    "weights": dict(zip(keys, weights)),
                    "iou_thr": iou_thr,
                    "skip_box_thr": skip,
                    "ap50": round(ap50, 5),
                    "n_preds": len(preds),
                    "dt": round(dt, 1),
                })
                done = len(results)
                elapsed = time.time() - t_global
                eta = (n_configs - done) * (elapsed / done) if done > 0 else 0
                print(f"[{done}/{n_configs}] weights={weights} iou={iou_thr} skip={skip} "
                      f"AP50={ap50:.4f} n={len(preds):,} ({dt:.1f}s, eta={eta:.0f}s)",
                      flush=True)

    results.sort(key=lambda r: -r["ap50"])
    args.out.write_text(json.dumps({
        "topk": results[:args.topk],
        "all": results,
        "config": {
            "models": args.models,
            "short_names": short_names,
            "iou_grid": iou_grid,
            "skip_grid": skip_grid,
            "stage": args.stage,
        },
    }, indent=2))
    print(f"\n[done] {len(results)} configs in {time.time()-t_global:.0f}s")
    print(f"[done] top-5 written to {args.out}")
    for r in results[:5]:
        print(f"  AP50={r['ap50']:.4f}  iou={r['iou_thr']} skip={r['skip_box_thr']}  weights={r['weights']}")


if __name__ == "__main__":
    main()
