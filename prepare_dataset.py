# ABOUTME: Master pipeline script for dataset preparation.
# ABOUTME: Runs: split -> oversample -> convert to YOLO -> generate yaml
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.data_split import stratified_split
from src.oversample import oversample_rare_classes
from src.generate_yaml import generate_yaml


BASE = Path(__file__).parent


def step(name: str):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")


def prepare_dataset(
    data_dir: Path = BASE / "data",
    val_size: float = 0.15,
    min_annotations: int = 50,
    seed: int = 42,
    skip_split: bool = False,
    skip_oversample: bool = False,
):
    t0 = time.time()

    annotations_path = data_dir / "train" / "annotations.json"
    images_dir = data_dir / "train" / "images"

    assert annotations_path.exists(), f"Not found: {annotations_path}"
    assert images_dir.exists(), f"Not found: {images_dir}"

    # 1. Split
    if not skip_split:
        step("Step 1: Stratified split (train/val)")
        stratified_split(
            annotations_path=annotations_path,
            images_dir=images_dir,
            output_dir=data_dir,
            val_size=val_size,
            seed=seed,
        )
    else:
        print("Step 1: Skipped (--skip-split)")

    # 2. Oversample
    if not skip_oversample:
        step("Step 2: Oversample rare classes")

        train_ann = data_dir / "train" / "annotations.json"
        train_images = data_dir / "train" / "images"

        # Если split уже создал data/train/ как выход — используем его
        split_train_ann = data_dir / "train" / "annotations.json"
        if split_train_ann.exists():
            train_ann = split_train_ann
            train_images = data_dir / "train" / "images"

        oversample_rare_classes(
            annotations_path=train_ann,
            images_dir=train_images,
            output_dir=data_dir / "train_balanced",
            min_annotations=min_annotations,
            seed=seed,
        )
    else:
        print("Step 2: Skipped (--skip-oversample)")

    # 3. Generate YAML
    step("Step 3: Generate bottles.yaml")
    generate_yaml()

    # Summary
    elapsed = time.time() - t0
    step("Done!")
    _print_summary(data_dir)
    print(f"\nTotal time: {elapsed:.1f}s")


def _print_summary(data_dir: Path):
    for split in ["train", "val", "train_balanced"]:
        ann_path = data_dir / split / "annotations.json"
        if not ann_path.exists():
            print(f"  {split}: not found")
            continue
        with open(ann_path) as f:
            coco = json.load(f)
        n_images = len(coco["images"])
        n_anns = len(coco["annotations"])
        n_cats = len(set(a["category_id"] for a in coco["annotations"]))
        print(f"  {split:>15}: {n_images:>5} images | {n_anns:>6} annotations | {n_cats:>3} classes")

    yaml_path = Path("bottles.yaml")
    if yaml_path.exists():
        print(f"\n  bottles.yaml: OK")
    else:
        print(f"\n  bottles.yaml: MISSING")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset preparation pipeline")
    parser.add_argument("--data-dir", type=Path, default=BASE / "data")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--min-annotations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-split", action="store_true")
    parser.add_argument("--skip-oversample", action="store_true")
    args = parser.parse_args()

    prepare_dataset(
        data_dir=args.data_dir,
        val_size=args.val_size,
        min_annotations=args.min_annotations,
        seed=args.seed,
        skip_split=args.skip_split,
        skip_oversample=args.skip_oversample,
    )