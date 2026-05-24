"""Build a repeat-factor-weighted training list for class-balanced fine-tune.

Reads YOLO-format labels under <data_root>/<split>/labels/, computes per-class
sample counts, assigns each image a weight = max over its classes of
1/sqrt(count(c)), then writes a flat list of image paths with each image
repeated floor(weight * K) times.

Adds optional FORCED boost for "sibling group" classes (the under-predicted
1L/12Yo-KAR classes from codex V3) so they appear extra often.

Output: train_<split>_cb_K{K}.txt with one absolute image path per line.

Usage:
    python scripts/build_balanced_trainlist.py \\
        --data-root /workspace/hackology2/data \\
        --split train_balanced \\
        --K 6 \\
        --boost-classes 279 143 156 \\
        --boost-factor 3 \\
        --out /workspace/hackology2/train_balanced_cb.txt

Then write a new YAML pointing `train: /abs/path/train_balanced_cb.txt`.
Ultralytics supports a txt list of image paths as the train source.
"""
from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path


def collect_classes(labels_dir: Path) -> tuple[dict[str, set[int]], Counter]:
    """Map image_stem -> {class_ids}, plus total class counts."""
    per_img: dict[str, set[int]] = {}
    counts: Counter = Counter()
    for lp in labels_dir.glob("*.txt"):
        classes = set()
        for line in lp.read_text().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            try:
                classes.add(int(parts[0]))
            except ValueError:
                continue
        per_img[lp.stem] = classes
        for c in classes:
            counts[c] += 1
    return per_img, counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", default="train_balanced")
    ap.add_argument("--K", type=int, default=6,
                    help="Repeat scale. Final per-img reps = max(1, round(weight * K))")
    ap.add_argument("--boost-classes", type=int, nargs="*", default=[],
                    help="YOLO 0-indexed class IDs to force-boost (multiply reps by --boost-factor)")
    ap.add_argument("--boost-factor", type=int, default=3)
    ap.add_argument("--max-reps", type=int, default=20,
                    help="Cap reps per image to avoid pathological inflation")
    ap.add_argument("--img-ext", default="jpg")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.data_root)
    labels_dir = root / args.split / "labels"
    images_dir = root / args.split / "images"
    if not labels_dir.exists():
        raise SystemExit(f"missing {labels_dir}")

    print(f"[scan] {labels_dir}")
    per_img, counts = collect_classes(labels_dir)
    n_imgs = len(per_img)
    n_classes = len(counts)
    print(f"[scan] {n_imgs} images, {n_classes} classes")

    # Rare-class weight: max over image's classes of 1/sqrt(count)
    weights = {}
    for stem, classes in per_img.items():
        if not classes:
            weights[stem] = 0.0
            continue
        weights[stem] = max(1.0 / math.sqrt(counts[c]) for c in classes)

    boost_set = set(args.boost_classes)
    lines: list[str] = []
    boost_hits = defaultdict(int)
    rep_hist = Counter()
    for stem, w in weights.items():
        if w == 0:
            continue
        img_path = images_dir / f"{stem}.{args.img_ext}"
        if not img_path.exists():
            # try png as fallback
            img_path = images_dir / f"{stem}.png"
            if not img_path.exists():
                continue
        reps = max(1, round(w * args.K))
        if boost_set & per_img[stem]:
            reps *= args.boost_factor
            for c in boost_set & per_img[stem]:
                boost_hits[c] += 1
        reps = min(reps, args.max_reps)
        rep_hist[reps] += 1
        for _ in range(reps):
            lines.append(str(img_path))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"[write] {len(lines):,} entries -> {out}")
    print(f"[reps hist] {dict(list(sorted(rep_hist.items()))[:20])}")
    print(f"[boost] hits by class: {dict(boost_hits)}")
    print(f"[rarest 5 classes] " + ", ".join(
        f"cls{c}:{counts[c]}" for c, _ in counts.most_common()[:-6:-1]))
    print(f"[most common 5]    " + ", ".join(
        f"cls{c}:{counts[c]}" for c, _ in counts.most_common(5)))


if __name__ == "__main__":
    main()
