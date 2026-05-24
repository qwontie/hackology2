"""Sibling-NMS postprocess (per Gemini error review, codex V4 alignment).

Problem: same physical bottle gets multiple high-conf predictions across
sibling SKUs (volume, age, idx1/idx2 variants, sometimes packaging). Standard
NMS is class-aware, so cross-class duplicates survive.

Fix: parse class names, build siblings on-the-fly, run per-image NMS-within-
group preserving highest-score pred.

Two passes:
  pass A (within-pack):  drop lower-score pred if IoU>0.6 AND siblings differ
                          only in {volume, idx1, idx2, age}  (same brand+pack)
  pass B (cross-pack):   drop lower-score pred if IoU>0.85 AND share brand
                          (BUT vs KAR/TUB on same physical object).

Both thresholds tunable from CLI.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

VOL_RE = re.compile(r"^\d+ml$")
AGE_RE = re.compile(r"^\d+Yo$")
IDX_RE = re.compile(r"^\d\d$")
PACKS = {"BUT", "KAR", "TUB", "MIN", "SZK"}


def parse_name(name: str) -> dict:
    """Extract structural fields. Robust to off-format names — unknown bits stay in 'extras'."""
    parts = name.split("_")
    vol = next((p for p in parts if VOL_RE.match(p)), None)
    age = next((p for p in parts if AGE_RE.match(p)), None)
    pack = next((p for p in parts if p in PACKS), None)
    brand = parts[0] if parts else "?"
    # core = everything except vol/age/idx — used for sibling check
    core = tuple(p for p in parts if not (VOL_RE.match(p) or AGE_RE.match(p) or IDX_RE.match(p)))
    return {"brand": brand, "vol": vol, "age": age, "pack": pack, "core": core, "parts": parts}


def iou_xywh(a: list[float], b: list[float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def apply_sibling_nms(
    preds: list[dict],
    cats: dict[int, str],
    iou_within: float = 0.60,
    iou_cross: float = 0.85,
    min_score_keep: float = 0.05,
) -> tuple[list[dict], dict]:
    """Return (filtered_preds, stats)."""
    cat_info = {cid: parse_name(name) for cid, name in cats.items()}

    by_img: dict[int, list[int]] = defaultdict(list)
    for i, p in enumerate(preds):
        by_img[p["image_id"]].append(i)

    suppressed = set()
    stats = {"within_pack_drops": 0, "cross_pack_drops": 0, "n_images": len(by_img)}

    for idxs in by_img.values():
        # Sort indices by score descending so we drop lower-score in ties
        idxs = sorted(idxs, key=lambda k: -preds[k]["score"])

        # Pass A: within-pack siblings
        # Group by (brand, pack, core_signature) and run per-group NMS
        for i, ki in enumerate(idxs):
            if ki in suppressed:
                continue
            pi = preds[ki]
            ci = cat_info.get(pi["category_id"])
            if ci is None:
                continue
            for kj in idxs[i + 1:]:
                if kj in suppressed:
                    continue
                pj = preds[kj]
                cj = cat_info.get(pj["category_id"])
                if cj is None:
                    continue
                if ci["brand"] != cj["brand"]:
                    continue
                # Same brand+pack with sibling fields differing -> Pass A
                if ci["pack"] == cj["pack"] and ci["core"] == cj["core"]:
                    if iou_xywh(pi["bbox"], pj["bbox"]) >= iou_within:
                        if pj["score"] >= min_score_keep or pi["score"] >= min_score_keep:
                            suppressed.add(kj)
                            stats["within_pack_drops"] += 1
                # Cross-pack same-brand (e.g. BUT vs KAR for same SKU)
                elif ci["pack"] != cj["pack"] and ci["pack"] is not None and cj["pack"] is not None:
                    if iou_xywh(pi["bbox"], pj["bbox"]) >= iou_cross:
                        if pj["score"] >= min_score_keep or pi["score"] >= min_score_keep:
                            suppressed.add(kj)
                            stats["cross_pack_drops"] += 1

    kept = [p for i, p in enumerate(preds) if i not in suppressed]
    stats["before"] = len(preds)
    stats["after"] = len(kept)
    stats["dropped_total"] = len(suppressed)
    return kept, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="Input COCO JSON")
    ap.add_argument("--meta", required=True, help="test_images.json (for categories)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iou-within", type=float, default=0.60)
    ap.add_argument("--iou-cross", type=float, default=0.85)
    ap.add_argument("--min-score-keep", type=float, default=0.05,
                    help="Only run NMS if at least one pred in pair has score >= this. "
                         "Skips trash-vs-trash work to save cycles.")
    args = ap.parse_args()

    preds = json.loads(Path(args.preds).read_text())
    meta = json.loads(Path(args.meta).read_text())
    cats = {c["id"]: c["name"] for c in meta["categories"]}
    print(f"[load] {len(preds):,} preds, {len(cats)} cats")

    kept, stats = apply_sibling_nms(
        preds, cats,
        iou_within=args.iou_within,
        iou_cross=args.iou_cross,
        min_score_keep=args.min_score_keep,
    )
    print(f"[nms] within_pack={stats['within_pack_drops']:,} cross_pack={stats['cross_pack_drops']:,} "
          f"-> {stats['before']:,} -> {stats['after']:,} (-{stats['dropped_total']:,})")

    Path(args.out).write_text(json.dumps(kept))
    print(f"[write] {args.out} ({Path(args.out).stat().st_size/1024/1024:.2f}MB)")


if __name__ == "__main__":
    main()
