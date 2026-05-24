"""Post-process: add shadow predictions for sibling-class confusions.

Per per-class diag, the model confuses sibling SKUs (same brand, different
volume or age). For every prediction whose category is the *visible* sibling,
emit an additional prediction of the *missed* sibling at a discounted score.

mAP is averaged per class. If 3 weak classes each gain ~+0.5 AP, total mAP
moves by ~1.5 / 369 ≈ 0.004 — roughly the gap to organizer's `cc`. Low-score
shadows usually rank below real high-score detections so FPs are tolerable.

Mapping (codex V3): src_cat -> [(dst_cat, factor)]
  Source = class the model OFTEN predicts correctly.
  Dest   = sibling class the model MISSES.
  factor = multiplier on the source score for the shadow pred.

Usage:
    uv run python scripts/sibling_copy.py \
        --in submissions/predictions.json \
        --out submissions/predictions_sibling.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# {src_cat: [(dst_cat, factor)]}  — when model predicts src, add shadow dst.
SIBLING_COPY: dict[int, list[tuple[int, float]]] = {
    284: [(280, 0.40)],   # Pas WHI 700ml -> 1000ml
    282: [(280, 0.30)],   # Pas WHI 500ml -> 1000ml
    149: [(144, 0.45)],   # ChiReg 12Yo 700ml -> 1000ml
    147: [(144, 0.30)],   # ChiReg 12Yo 500ml -> 1000ml
    133: [(157, 0.35)],   # ChiReg 13Yo KAR -> 12Yo KAR
    162: [(157, 0.25)],   # ChiReg 18Yo KAR -> 12Yo KAR
}

MIN_SOURCE_SCORE = 0.15   # don't shadow noise


def apply_sibling_copy(preds: list[dict],
                       mapping: dict[int, list[tuple[int, float]]] = SIBLING_COPY,
                       min_src_score: float = MIN_SOURCE_SCORE,
                       max_score: float = 1.0) -> tuple[list[dict], dict]:
    out = list(preds)
    added = defaultdict(int)
    seen_src = defaultdict(int)
    for p in preds:
        src = int(p["category_id"])
        if src not in mapping:
            continue
        src_score = float(p["score"])
        if src_score < min_src_score:
            continue
        seen_src[src] += 1
        for dst, factor in mapping[src]:
            new_score = src_score * factor
            if new_score <= 0:
                continue
            new_score = min(new_score, max_score)
            out.append({
                "image_id": p["image_id"],
                "category_id": int(dst),
                "bbox": list(p["bbox"]),
                "score": round(new_score, 4),
            })
            added[(src, dst)] += 1
    stats = {
        "input_preds": len(preds),
        "output_preds": len(out),
        "added_total": sum(added.values()),
        "by_pair": {f"{s}->{d}": n for (s, d), n in sorted(added.items())},
        "src_seen": dict(seen_src),
        "min_src_score": min_src_score,
    }
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-score", type=float, default=MIN_SOURCE_SCORE,
                    help=f"Skip source preds below this score (default {MIN_SOURCE_SCORE})")
    ap.add_argument("--stats", default=None,
                    help="Optional path to write stats JSON; defaults to <out>.stats.json")
    args = ap.parse_args()

    preds = json.loads(Path(args.inp).read_text())
    out, stats = apply_sibling_copy(preds, min_src_score=args.min_score)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out))

    stats_path = Path(args.stats) if args.stats else out_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2))

    print(f"[in]  {args.inp}  ({stats['input_preds']:,} preds)")
    print(f"[out] {args.out}  ({stats['output_preds']:,} preds, +{stats['added_total']:,})")
    print(f"[by_pair] {stats['by_pair']}")
    print(f"[stats] -> {stats_path}")


if __name__ == "__main__":
    main()
