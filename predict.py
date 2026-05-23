# ABOUTME: Baseline prediction script using ultralytics YOLOv8n for object detection.
# ABOUTME: CLI interface: predict --input DIR --output predictions.json. Outputs COCO-format predictions.
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ultralytics import YOLO


def load_taxonomy(path: Path = Path("taxonomy.json")) -> dict[int, int]:
    """Load category mapping from taxonomy.json.

    Returns a mapping from COCO-80 class index to hackathon category ID.
    If taxonomy.json is absent, uses identity mapping.
    """
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    # Map from position index to category ID
    return {i: cat["id"] for i, cat in enumerate(data["categories"])}


def load_image_id_map(annotations_path: Path) -> dict[str, int]:
    """Load mapping from image filename to image_id from annotations.json.

    Args:
        annotations_path: Path to COCO-format annotations JSON.

    Returns:
        Dict mapping filename (e.g. "img001.jpg") to integer image_id.
    """
    if not annotations_path.exists():
        print(f"WARNING: {annotations_path} not found, image_id mapping unavailable", file=sys.stderr)
        return {}
    data = json.loads(annotations_path.read_text(encoding="utf-8"))
    return {img["file_name"]: img["id"] for img in data["images"]}


def predict_directory(
    input_dir: Path,
    model_path: str = "yolov8n.pt",
    confidence: float = 0.25,
    taxonomy_path: Path = Path("taxonomy.json"),
    annotations_path: Path = Path("test_images.json"),
) -> list[dict]:
    """Run YOLOv8n inference on all images in a directory.

    Args:
        input_dir: Directory containing input images.
        model_path: Path to YOLO model weights (default: pretrained yolov8n).
        confidence: Minimum confidence threshold.
        taxonomy_path: Path to taxonomy.json for category mapping.
        annotations_path: Path to annotations.json for image_id mapping.

    Returns:
        List of COCO-format prediction dicts.
    """
    model = YOLO(model_path)
    cat_map = load_taxonomy(taxonomy_path)
    filename_to_id = load_image_id_map(annotations_path)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in image_extensions
    )

    if not image_files:
        print(f"WARNING: No images found in {input_dir}", file=sys.stderr)
        return []

    predictions: list[dict] = []

    for img_path in image_files:
        image_id = filename_to_id.get(img_path.name)
        if image_id is None:
            print(f"WARNING: {img_path.name} not in annotations, skipping", file=sys.stderr)
            continue

        results = model(str(img_path), conf=confidence, verbose=False)

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                # YOLO outputs xyxy, convert to COCO format [x, y, w, h]
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                w = x2 - x1
                h = y2 - y1

                cls_id = int(boxes.cls[i].item())
                category_id = cat_map.get(cls_id)
                if category_id is None:
                    continue  # skip predictions outside our taxonomy
                score = float(boxes.conf[i].item())

                predictions.append({
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                    "score": round(score, 4),
                })

    return predictions


def main() -> None:
    """CLI entry point: predict --input DIR --output predictions.json"""
    parser = argparse.ArgumentParser(
        description="Run object detection and produce COCO-format predictions"
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Directory containing input images",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("predictions.json"),
        help="Output path for predictions JSON (default: predictions.json)",
    )
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt",
        help="Path to YOLO model weights (default: yolov8n.pt pretrained)",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.25,
        help="Minimum confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--taxonomy", type=Path, default=Path("taxonomy.json"),
        help="Path to taxonomy.json for category mapping",
    )
    parser.add_argument(
        "--annotations", type=Path, default=Path("test_images.json"),
        help="Path to test_images.json for image_id mapping (default: test_images.json)",
    )
    args = parser.parse_args()

    if not args.input.is_dir():
        print(f"ERROR: {args.input} is not a directory", file=sys.stderr)
        sys.exit(1)

    predictions = predict_directory(
        input_dir=args.input,
        model_path=args.model,
        confidence=args.confidence,
        taxonomy_path=args.taxonomy,
        annotations_path=args.annotations,
    )

    args.output.write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
