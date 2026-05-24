"""One-shot uploader: pushes champion ensemble weights to Hugging Face Hub.

Usage:
    # First time setup (one of):
    huggingface-cli login                                 # interactive
    export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx              # CI/non-interactive

    # Upload:
    uv run python scripts/upload_hf_weights.py
    # or override:
    HF_REPO=qwontie/bottle-detector-weights uv run python scripts/upload_hf_weights.py

Repo must exist and be PUBLIC (so predict.py downloads without auth).
Create via web UI at https://huggingface.co/new or:
    huggingface-cli repo create bottle-detector-weights --type model --organization qwontie
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, login
except ImportError:
    print("ERROR: pip install huggingface_hub", file=sys.stderr)
    sys.exit(1)


REPO = os.environ.get("HF_REPO", "qwontie/bottle-detector-weights")
ROOT = Path(__file__).resolve().parent.parent

# Local path -> filename on HF. Local layout is a mess (split across _weights/
# and data_probe/weights/new_models/) — flatten on upload.
FILES: dict[Path, str] = {
    # 5-model champion ensemble (public LB 0.8172)
    ROOT / "_weights" / "student_m_1536_cwd_best.pt":               "student_m_1536_cwd_best.pt",
    ROOT / "_weights" / "teacher_x_1536_all_best.pt":               "teacher_x_1536_all_best.pt",
    ROOT / "data_probe/weights/new_models/y11l_1536_best.pt":       "y11l_1536_best.pt",
    ROOT / "data_probe/weights/new_models/yolo11x_v2_best.pt":      "yolo11x_v2_best.pt",
    ROOT / "data_probe/weights/new_models/yolov8x_cb_best.pt":      "yolov8x_cb_best.pt",
}


def main() -> int:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        login(token=tok, add_to_git_credential=False)

    api = HfApi()
    missing = [str(p) for p in FILES if not p.exists()]
    if missing:
        print("ERROR: missing local weights:\n  " + "\n  ".join(missing), file=sys.stderr)
        return 1

    print(f"Uploading {len(FILES)} weights -> https://huggingface.co/{REPO}")
    for local, remote in FILES.items():
        size_mb = local.stat().st_size / 1024**2
        print(f"  {remote}  ({size_mb:.1f} MB)")
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=REPO,
            repo_type="model",
            commit_message=f"upload {remote}",
        )
    print("DONE")
    print(f"Direct URL pattern: https://huggingface.co/{REPO}/resolve/main/<filename>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
