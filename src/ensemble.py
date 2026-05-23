import json
from pathlib import Path

from ensemble_boxes import weighted_boxes_fusion


def load_predictions(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_image_info(test_images_path: Path) -> dict[int, tuple[int, int]]:
    data = json.loads(test_images_path.read_text(encoding="utf-8"))
    images = data if isinstance(data, list) else data.get("images", [])
    return {img["id"]: (img["width"], img["height"]) for img in images}


def wbf_ensemble(
    prediction_paths: list[Path],
    test_images_path: Path,
    iou_thr: float = 0.5,
    skip_box_thr: float = 0.05,
    weights: list[float] | None = None,
) -> list[dict]:
    all_preds = [load_predictions(p) for p in prediction_paths]
    image_info = load_image_info(test_images_path)

    if weights is None:
        weights = [1.0] * len(all_preds)

    # Group by image_id
    from collections import defaultdict
    preds_by_image: dict[int, list[list[dict]]] = defaultdict(lambda: [[] for _ in all_preds])
    for model_idx, model_preds in enumerate(all_preds):
        for pred in model_preds:
            preds_by_image[pred["image_id"]][model_idx].append(pred)

    fused: list[dict] = []

    for image_id, per_model in preds_by_image.items():
        if image_id not in image_info:
            continue
        img_w, img_h = image_info[image_id]

        boxes_list, scores_list, labels_list = [], [], []

        for model_preds in per_model:
            boxes, scores, labels = [], [], []
            for pred in model_preds:
                x, y, w, h = pred["bbox"]
                # normalize to [0, 1] for WBF
                x1 = max(0.0, x / img_w)
                y1 = max(0.0, y / img_h)
                x2 = min(1.0, (x + w) / img_w)
                y2 = min(1.0, (y + h) / img_h)
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([x1, y1, x2, y2])
                scores.append(pred["score"])
                labels.append(pred["category_id"])
            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)

        if not any(boxes_list):
            continue

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
        )

        for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
            x1, y1, x2, y2 = box
            fused.append({
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [
                    round(x1 * img_w, 2),
                    round(y1 * img_h, 2),
                    round((x2 - x1) * img_w, 2),
                    round((y2 - y1) * img_h, 2),
                ],
                "score": round(float(score), 4),
            })

    return fused


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WBF ensemble over multiple predictions.json")
    parser.add_argument("--preds", nargs="+", type=Path, required=True,
                        help="Paths to predictions.json from each model")
    parser.add_argument("--test-images", type=Path, default=Path("test_images.json"))
    parser.add_argument("--output", type=Path, default=Path("submissions/predictions.json"))
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--skip-box-thr", type=float, default=0.05)
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    args = parser.parse_args()

    result = wbf_ensemble(
        prediction_paths=args.preds,
        test_images_path=args.test_images,
        iou_thr=args.iou_thr,
        skip_box_thr=args.skip_box_thr,
        weights=args.weights,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Fused {len(result)} predictions -> {args.output}")