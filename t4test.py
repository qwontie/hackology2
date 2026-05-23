"""Self-contained T4 profiling script for predict_v2.

Run on a fresh T4 box after:
    git clone https://github.com/qwontie/hackology2 && cd hackology2
    uv sync --frozen

Then:
    uv run python t4test.py                       # all 4 levels
    uv run python t4test.py --levels L1 L2        # only specific levels
    uv run python t4test.py --data-url '<url>'    # override fuckingfast URL

Does:
  1. nvidia-smi sanity
  2. Downloads public_test.tar.gz (464MB) from fuckingfast → /tmp/public_test/
  3. Runs predict_v2.py at each level on 481 imgs, captures wallclock
  4. Prints summary table; saves output JSONs to /tmp/L{1..4}_*.json

We're profiling the eval pipeline that runs at finals: T4 16GB, 30min budget.
Goal: pick the heaviest level that lands under ~25min (5min safety margin).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# fuckingfast direct-download URL (expires ~3 days after upload — override with --data-url if 410'd)
DEFAULT_DATA_URL = "https://ts.fuckingfast.net/d/psw4xpos2kok?v=t_kO4BNctek5QB7RejF24kqktnHjkpTbqdj-JEvojSlhp4zeFnC4WRQMzOWC0JLB15XsF_bpsTqaEjLFDHWLFdAmkHpEFCTNeBfGCSKJGMG8KGTVz5pvyyn97hnUcP5uDGvqDQFqZAriLH4bWy9yBu3YZYuI4qo"

DATA_TARBALL = Path("/tmp/public_test.tar.gz")
DATA_DIR = Path("/tmp/public_test")            # after extraction: /tmp/public_test/images/, /tmp/public_test/test_images.json
IMAGES_DIR = DATA_DIR / "images"
ANNOTATIONS = DATA_DIR / "test_images.json"
EXPECTED_IMG_COUNT = 481


LEVELS = {
    "L1": {
        "label": "L1 student balanced (1536 + flip)",
        "args": ["--mode", "balanced", "--model", "student"],
        "out":  "/tmp/L1_student_balanced.json",
    },
    "L2": {
        "label": "L2 student heavy (multi-scale + flip + WBF)",
        "args": ["--mode", "heavy", "--model", "student"],
        "out":  "/tmp/L2_student_heavy.json",
    },
    "L3": {
        "label": "L3 ensemble balanced (student+teacher, 1536 + flip)",
        "args": ["--mode", "balanced", "--models", "student", "teacher"],
        "out":  "/tmp/L3_ensemble_balanced.json",
    },
    "L4": {
        "label": "L4 ensemble heavy (student+teacher multi-scale + flip + WBF)",
        "args": ["--mode", "heavy", "--models", "student", "teacher"],
        "out":  "/tmp/L4_ensemble_heavy.json",
    },
}


def banner(msg: str) -> None:
    print(f"\n{'='*72}\n {msg}\n{'='*72}", flush=True)


def run(cmd: list[str], check: bool = True) -> int:
    """Run a command, stream its output, return the exit code."""
    print(f"$ {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd)
    if check and rc != 0:
        print(f"!! command failed with rc={rc}", file=sys.stderr)
    return rc


def show_gpu() -> None:
    banner("0. GPU check")
    if shutil.which("nvidia-smi"):
        subprocess.call(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv"])
    else:
        print("WARNING: nvidia-smi not found — are we on a GPU box?", file=sys.stderr)


def download_data(url: str) -> None:
    banner("1. Download + extract public_test")
    if IMAGES_DIR.exists():
        n = len(list(IMAGES_DIR.glob("*.jpg")))
        if n >= EXPECTED_IMG_COUNT:
            print(f"  already extracted: {n} images in {IMAGES_DIR}")
            return
        print(f"  partial extraction ({n}/{EXPECTED_IMG_COUNT}), redoing")
        shutil.rmtree(DATA_DIR)

    if not DATA_TARBALL.exists():
        print(f"  downloading {url[:80]}...")
        t0 = time.time()
        urllib.request.urlretrieve(url, str(DATA_TARBALL))
        size_mb = DATA_TARBALL.stat().st_size / 1024**2
        print(f"  -> {DATA_TARBALL} ({size_mb:.1f}MB in {time.time()-t0:.1f}s)")
    else:
        print(f"  cached tarball: {DATA_TARBALL} ({DATA_TARBALL.stat().st_size / 1024**2:.1f}MB)")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # The tarball was created with `tar czf ... public_test/` so it contains a top-level "public_test/" dir.
    # We strip it so contents land directly in DATA_DIR.
    run(["tar", "xzf", str(DATA_TARBALL), "-C", str(DATA_DIR.parent)])

    if not IMAGES_DIR.exists():
        # Fallback: tar may have a different top level — try to find images dir
        candidates = list(DATA_DIR.parent.rglob("test_images.json"))
        if candidates:
            found_root = candidates[0].parent
            if found_root != DATA_DIR:
                print(f"  remapping {found_root} -> {DATA_DIR}")
                if DATA_DIR.exists():
                    shutil.rmtree(DATA_DIR)
                shutil.move(str(found_root), str(DATA_DIR))

    n = len(list(IMAGES_DIR.glob("*.jpg"))) if IMAGES_DIR.exists() else 0
    print(f"  -> {n} images, annotations: {ANNOTATIONS} (exists={ANNOTATIONS.exists()})")
    assert n == EXPECTED_IMG_COUNT, f"expected {EXPECTED_IMG_COUNT} imgs, got {n}"
    assert ANNOTATIONS.exists(), f"missing {ANNOTATIONS}"


def warmup() -> None:
    """Pre-download student weights + warm CUDA on 20 imgs so per-level timings are fair."""
    banner("2. Warmup (20 imgs, downloads weights + JITs CUDA)")
    warm_dir = Path("/tmp/warmup_imgs")
    warm_dir.mkdir(exist_ok=True)
    for p in sorted(IMAGES_DIR.glob("*.jpg"))[:20]:
        link = warm_dir / p.name
        if not link.exists():
            link.symlink_to(p)
    cmd = [
        sys.executable, "predict_v2.py",
        "--input", str(warm_dir),
        "--annotations", str(ANNOTATIONS),
        "--output", "/tmp/warmup.json",
        "--mode", "balanced",
        "--model", "student",
    ]
    t0 = time.time()
    rc = run(cmd, check=False)
    print(f"  warmup done in {time.time()-t0:.1f}s (rc={rc})")


def run_level(key: str) -> dict:
    """Run one predict_v2 level, capture wallclock + output size."""
    spec = LEVELS[key]
    banner(f"3. {spec['label']}")
    cmd = [
        sys.executable, "predict_v2.py",
        "--input", str(IMAGES_DIR),
        "--annotations", str(ANNOTATIONS),
        "--output", spec["out"],
        *spec["args"],
    ]
    t0 = time.time()
    rc = run(cmd, check=False)
    elapsed = time.time() - t0

    result = {"key": key, "label": spec["label"], "elapsed_s": elapsed, "rc": rc}
    out = Path(spec["out"])
    if out.exists():
        try:
            preds = json.loads(out.read_text())
            result["n_preds"] = len(preds)
            result["n_imgs"] = len(set(p["image_id"] for p in preds))
            result["n_cats"] = len(set(p["category_id"] for p in preds))
            result["size_mb"] = out.stat().st_size / 1024**2
        except Exception as e:
            result["error"] = f"parse output failed: {e}"
    else:
        result["error"] = "output file missing"

    print(f"  -> elapsed={elapsed:.1f}s  rc={rc}  "
          f"preds={result.get('n_preds', '?')}  "
          f"imgs={result.get('n_imgs', '?')}  "
          f"cats={result.get('n_cats', '?')}")
    return result


def print_summary(results: list[dict]) -> None:
    banner("4. Summary")
    budget = 30 * 60   # T4 finals budget in seconds
    safe   = 25 * 60   # our safety margin
    print(f"  (budget = {budget}s = 30min, our target = ≤{safe}s = 25min)")
    print()
    print(f"  {'Level':40s} {'Time':>8s}  {'Verdict':12s} {'Preds':>7s} {'Imgs':>5s} {'Cats':>5s}")
    print(f"  {'-'*40} {'-'*8}  {'-'*12} {'-'*7} {'-'*5} {'-'*5}")
    for r in results:
        t = r["elapsed_s"]
        if "error" in r:
            verdict = "ERROR"
        elif t > budget:
            verdict = "❌ OVER"
        elif t > safe:
            verdict = "⚠️  RISKY"
        else:
            verdict = "✅ FITS"
        print(f"  {r['label']:40s} {t:>7.0f}s  {verdict:12s} "
              f"{r.get('n_preds', '-'):>7} {r.get('n_imgs', '-'):>5} {r.get('n_cats', '-'):>5}")
    print()
    fitting = [r for r in results if r.get("elapsed_s", 9e9) <= safe and "error" not in r]
    if fitting:
        # pick the highest level that fits (assumes L1<L2<L3<L4 ordering of quality)
        winner = max(fitting, key=lambda r: r["key"])
        print(f"  → CHAMPION: {winner['label']}  ({winner['elapsed_s']:.0f}s, output {winner.get('size_mb', 0):.1f}MB)")
        print(f"    cp {LEVELS[winner['key']]['out']} submissions/predictions.json && git tag final && git push --tags")
    else:
        print("  → NO LEVEL FITS within safety budget. Try --mode safe --model student manually.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--levels", nargs="+", default=list(LEVELS.keys()),
                   help=f"Which levels to run, default all: {list(LEVELS.keys())}")
    p.add_argument("--data-url", default=DEFAULT_DATA_URL,
                   help="Direct download URL for public_test.tar.gz (fuckingfast link expires in ~3 days)")
    p.add_argument("--skip-warmup", action="store_true",
                   help="Skip the 20-img warmup pass (timings will include weight downloads + CUDA JIT)")
    args = p.parse_args()

    unknown = [l for l in args.levels if l not in LEVELS]
    if unknown:
        print(f"unknown levels: {unknown}, valid: {list(LEVELS.keys())}", file=sys.stderr)
        sys.exit(1)

    show_gpu()
    download_data(args.data_url)
    if not args.skip_warmup:
        warmup()

    results = []
    for key in args.levels:
        try:
            results.append(run_level(key))
        except KeyboardInterrupt:
            print(f"\n  interrupted during {key}, stopping", file=sys.stderr)
            break

    print_summary(results)


if __name__ == "__main__":
    main()
