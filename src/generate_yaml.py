import json
import yaml
from pathlib import Path


def generate_yaml():
    base = Path(__file__).parent.parent

    with open(base / "taxonomy.json") as f:
        taxonomy = json.load(f)

    cat_ids = sorted(c["id"] for c in taxonomy["categories"])
    names = {i: c["name"] for i, c in enumerate(
        sorted(taxonomy["categories"], key=lambda x: x["id"])
    )}

    bottles_yaml = {
        "path": str(base / "data"),
        "train": "train_balanced/images",
        "val": "val/images",
        "test": "public_test/images",
        "nc": len(names),
        "names": names,
    }

    with open(base / "bottles.yaml", "w") as f:
        yaml.dump(bottles_yaml, f, allow_unicode=True)

    print(f"Ready: {len(names)} classes")