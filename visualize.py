# ABOUTME: Visualizes top-10 best and worst predictions from predictions.json.
# ABOUTME: Saves annotated images locally with bounding boxes and scores.
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image


def load_data(
    predictions_path: Path,
    test_images_path: Path,
    taxonomy_path: Path,
) -> tuple[list[dict], dict[int, dict], dict[int, str]]:
    preds = json.loads(predictions_path.read_text(encoding="utf-8"))
    test_data = json.loads(test_images_path.read_text(encoding="utf-8"))
    images = test_data if isinstance(test_data, list) else test_data.get("images", [])
    img_map = {img["id"]: img for img in images}

    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    cat_names = {c["id"]: c["name"] for c in taxonomy["categories"]}

    return preds, img_map, cat_names


def pick_images(preds: list[dict], n: int = 10) -> tuple[list[int], list[int]]:
    """Pick image_ids with highest and lowest avg score."""
    scores_by_image: dict[int, list[float]] = defaultdict(list)
    for pred in preds:
        scores_by_image[pred["image_id"]].append(pred["score"])

    avg_scores = {img_id: sum(s) / len(s) for img_id, s in scores_by_image.items()}
    sorted_ids = sorted(avg_scores, key=avg_scores.get)

    worst = sorted_ids[:n]
    best = sorted_ids[-n:][::-1]
    return best, worst


def draw_image(
    img_path: Path,
    preds: list[dict],
    cat_names: dict[int, str],
    title: str,
    output_path: Path,
) -> None:
    if not img_path.exists():
        print(f"SKIP: {img_path} not found")
        return

    fig, ax = plt.subplots(1, figsize=(14, 10))
    img = Image.open(img_path)
    ax.imshow(img)

    colors = plt.cm.Set3.colors

    for pred in preds:
        x, y, w, h = pred["bbox"]
        cat_id = pred["category_id"]
        score = pred["score"]
        cat_name = cat_names.get(cat_id, str(cat_id))
        short_name = cat_name.split("_")[0]  # только бренд для читаемости

        color = colors[cat_id % len(colors)]
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            x, y - 5,
            f"{short_name} {score:.2f}",
            fontsize=7, color=color,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7),
        )

    avg_score = sum(p["score"] for p in preds) / len(preds) if preds else 0
    ax.set_title(f"{title}\n{img_path.name} — {len(preds)} preds, avg score={avg_score:.3f}", fontsize=10)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def visualize(
    predictions_path: Path,
    images_dir: Path,
    test_images_path: Path,
    taxonomy_path: Path,
    output_dir: Path,
    n: int = 10,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    preds, img_map, cat_names = load_data(predictions_path, test_images_path, taxonomy_path)

    preds_by_image: dict[int, list[dict]] = defaultdict(list)
    for pred in preds:
        preds_by_image[pred["image_id"]].append(pred)

    best_ids, worst_ids = pick_images(preds, n=n)

    for rank, img_id in enumerate(best_ids, 1):
        img_info = img_map.get(img_id)
        if not img_info:
            continue
        img_path = images_dir / img_info["file_name"]
        img_preds = preds_by_image[img_id]
        out = output_dir / f"best_{rank:02d}_img{img_id}.jpg"
        draw_image(img_path, img_preds, cat_names, f"BEST #{rank}", out)

    for rank, img_id in enumerate(worst_ids, 1):
        img_info = img_map.get(img_id)
        if not img_info:
            continue
        img_path = images_dir / img_info["file_name"]
        img_preds = preds_by_image[img_id]
        out = output_dir / f"worst_{rank:02d}_img{img_id}.jpg"
        draw_image(img_path, img_preds, cat_names, f"WORST #{rank}", out)

    print(f"\nDone. Saved {n*2} images to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize best/worst predictions")
    parser.add_argument("--predictions", type=Path, default=Path("submissions/predictions.json"))
    parser.add_argument("--images-dir", type=Path, default=Path("data/public_test/images"))
    parser.add_argument("--test-images", type=Path, default=Path("test_images.json"))
    parser.add_argument("--taxonomy", type=Path, default=Path("taxonomy.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("visualizations"))
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    visualize(
        predictions_path=args.predictions,
        images_dir=args.images_dir,
        test_images_path=args.test_images,
        taxonomy_path=args.taxonomy,
        output_dir=args.output_dir,
        n=args.n,
    )