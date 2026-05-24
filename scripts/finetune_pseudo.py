"""Fine-tune student on real train + pseudo-labeled public_test.

Approach:
  - Start from student_m_1536_cwd_best.pt (our current best)
  - Combined dataset: original train.txt + pseudo train_pseudo.txt
  - Low LR (lr0=1e-4), small epochs (default 4)
  - No mosaic/mixup (we want to learn the test-domain distribution clean)
  - Keep imgsz=1536 to match deployment

Why this might help:
  - public_test (and hopefully private holdout) has its own distribution shift
    (lighting, ratio, angle) that the training data doesn't fully cover.
  - Pseudo-labels from a strong ensemble (L4) are noisy but bias-correct enough
    to nudge the student toward this distribution without breaking baseline.

Run example (on H100 #2):
  python finetune_pseudo.py \\
      --base-weights _weights/student_m_1536_cwd_best.pt \\
      --orig-data bottles.yaml \\
      --pseudo-train /tmp/pseudo_data/train_pseudo.txt \\
      --pseudo-yaml /tmp/bottles_pseudo.yaml \\
      --output runs/student_m_pseudo \\
      --epochs 4 --lr0 1e-4 --batch 16
"""
from __future__ import annotations

import argparse
import yaml
from pathlib import Path


def build_pseudo_yaml(orig: Path, pseudo_train: Path, out: Path) -> Path:
    with open(orig) as f:
        cfg = yaml.safe_load(f)

    # YOLO supports `train` as a list of source files. Keep original train + add pseudo.
    orig_train = cfg.get("train")
    if isinstance(orig_train, str):
        orig_train_list = [orig_train]
    elif isinstance(orig_train, list):
        orig_train_list = list(orig_train)
    else:
        raise ValueError(f"orig bottles.yaml has unexpected train: {orig_train!r}")

    # Resolve original train paths relative to bottles.yaml's `path` (if any) so
    # they remain valid when we drop the new yaml elsewhere.
    base_path = cfg.get("path")
    if base_path:
        base = Path(base_path).resolve()
        orig_train_list = [str((base / t).resolve()) if not Path(t).is_absolute() else t
                           for t in orig_train_list]

    new_train = orig_train_list + [str(pseudo_train.resolve())]
    cfg["train"] = new_train
    cfg.pop("path", None)   # already absolutized

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-weights", type=Path, required=True,
                   help="path to student best.pt to fine-tune from")
    p.add_argument("--orig-data", type=Path, required=True,
                   help="original bottles.yaml")
    p.add_argument("--pseudo-train", type=Path, required=True,
                   help="train_pseudo.txt produced by pseudo_label.py")
    p.add_argument("--pseudo-yaml", type=Path, default=Path("/tmp/bottles_pseudo.yaml"),
                   help="where to write the combined data yaml")
    p.add_argument("--output", type=Path, default=Path("runs/student_m_pseudo"),
                   help="ultralytics project/name; will be `runs/student_m_pseudo`")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr0", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=1536)
    p.add_argument("--device", default="0")
    args = p.parse_args()

    yaml_path = build_pseudo_yaml(args.orig_data, args.pseudo_train, args.pseudo_yaml)
    print(f"  wrote combined yaml: {yaml_path}")
    print(f"  train sources: {open(yaml_path).read()}")

    from ultralytics import YOLO   # noqa: heavy import deferred
    model = YOLO(str(args.base_weights))

    project = args.output.parent
    name = args.output.name

    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        lr0=args.lr0,
        lrf=0.1,           # cosine final LR factor
        # Disable strong aug — we want to learn distribution shift cleanly:
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        # Keep modest geometric aug:
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015, hsv_s=0.4, hsv_v=0.3,
        degrees=0.0, translate=0.05, scale=0.2, shear=0.0,
        # Optimizer:
        optimizer="AdamW",
        warmup_epochs=0.5,
        weight_decay=0.0005,
        # Bookkeeping:
        project=str(project),
        name=name,
        save=True,
        save_period=-1,    # save only best+last
        plots=False,       # save time
        verbose=True,
        seed=42,
    )

    print(f"\n=== fine-tune done ===")
    best = Path(args.output) / "weights" / "best.pt"
    last = Path(args.output) / "weights" / "last.pt"
    print(f"  best: {best}  exists={best.exists()}")
    print(f"  last: {last}  exists={last.exists()}")


if __name__ == "__main__":
    main()
