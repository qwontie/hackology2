# usage

# uv run src/calibration.py \
#   --predictions submissions/predictions_wbf.json \
#   --val-predictions submissions/val_predictions.json \
#   --val-annotations data/val/annotations.json \
#   --output submissions/predictions.json



from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def compute_temperature(
    predictions: list[dict],
    ground_truth: list[dict],
    iou_thr: float = 0.5,
) -> float:
    """
    Fit temperature T via grid search on val predictions vs ground truth.

    T > 1.0 -> model overestimates confidence (scale down)
    T < 1.0 -> model underestimates confidence (scale up)

    Args:
        predictions: COCO-format predictions on val set
        ground_truth: COCO-format annotations (val annotations.json)
        iou_thr: IoU threshold for matching pred to GT

    Returns:
        Optimal temperature T
    """
    # Build GT lookup: image_id -> list of boxes
    from collections import defaultdict
    gt_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in ground_truth:
        gt_by_image[ann["image_id"]].append(ann)

    # For each prediction determine if it's TP or FP
    scores = []
    is_tp = []

    pred_by_image: dict[int, list[dict]] = defaultdict(list)
    for pred in predictions:
        pred_by_image[pred["image_id"]].append(pred)

    for image_id, preds in pred_by_image.items():
        gts = gt_by_image.get(image_id, [])
        matched_gt = set()

        # Sort by score desc
        preds_sorted = sorted(preds, key=lambda x: x["score"], reverse=True)

        for pred in preds_sorted:
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gts):
                if gt_idx in matched_gt:
                    continue
                if gt["category_id"] != pred["category_id"]:
                    continue
                iou = _iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            scores.append(pred["score"])
            if best_iou >= iou_thr and best_gt_idx >= 0:
                is_tp.append(1.0)
                matched_gt.add(best_gt_idx)
            else:
                is_tp.append(0.0)

    scores = np.array(scores)
    is_tp = np.array(is_tp)

    if len(scores) == 0:
        return 1.0

    # Grid search over T
    best_t = 1.0
    best_loss = float("inf")

    for t in np.linspace(0.1, 3.0, 300):
        calibrated = np.clip(scores ** (1.0 / t), 1e-7, 1 - 1e-7)
        # Binary cross-entropy
        loss = -np.mean(is_tp * np.log(calibrated) + (1 - is_tp) * np.log(1 - calibrated))
        if loss < best_loss:
            best_loss = loss
            best_t = t

    print(f"Best T={best_t:.4f} (loss={best_loss:.4f})")
    print(f"  T>1 means model was overconfident, T<1 means underconfident")
    return float(best_t)


def apply_temperature(predictions: list[dict], temperature: float) -> list[dict]:
    """Apply temperature scaling to predictions scores."""
    if abs(temperature - 1.0) < 1e-6:
        return predictions
    result = []
    for pred in predictions:
        calibrated_score = float(pred["score"] ** (1.0 / temperature))
        calibrated_score = max(1e-7, min(1 - 1e-7, calibrated_score))
        result.append({**pred, "score": round(calibrated_score, 4)})
    return result


def _iou(bbox1: list[float], bbox2: list[float]) -> float:
    """Compute IoU between two COCO bboxes [x, y, w, h]."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    ix1 = max(x1, x2)
    iy1 = max(y1, y2)
    ix2 = min(x1 + w1, x2 + w2)
    iy2 = min(y1 + h1, y2 + h2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = w1 * h1 + w2 * h2 - inter

    return inter / union if union > 0 else 0.0


def calibrate_predictions(
    predictions_path: Path,
    val_predictions_path: Path,
    val_annotations_path: Path,
    output_path: Path,
) -> float:
    """
    Full pipeline: fit T on val set, apply to test predictions.

    Args:
        predictions_path: test predictions to calibrate
        val_predictions_path: model predictions on val set
        val_annotations_path: ground truth annotations for val set
        output_path: where to save calibrated predictions
    Returns:
        temperature T that was applied
    """
    test_preds = json.loads(predictions_path.read_text(encoding="utf-8"))
    val_preds = json.loads(val_predictions_path.read_text(encoding="utf-8"))

    with open(val_annotations_path) as f:
        val_coco = json.load(f)
    val_gt = val_coco["annotations"]

    T = compute_temperature(val_preds, val_gt)
    calibrated = apply_temperature(test_preds, T)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(calibrated, indent=2) + "\n", encoding="utf-8")
    print(f"Calibrated {len(calibrated)} predictions with T={T:.4f} -> {output_path}")
    return T


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True,
                        help="Test predictions to calibrate")
    parser.add_argument("--val-predictions", type=Path, required=True,
                        help="Model predictions on val set")
    parser.add_argument("--val-annotations", type=Path, default=Path("data/val/annotations.json"))
    parser.add_argument("--output", type=Path, default=Path("submissions/predictions_calibrated.json"))
    args = parser.parse_args()

    calibrate_predictions(
        predictions_path=args.predictions,
        val_predictions_path=args.val_predictions,
        val_annotations_path=args.val_annotations,
        output_path=args.output,
    )