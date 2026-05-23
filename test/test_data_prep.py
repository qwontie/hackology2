import json
from pathlib import Path

from src.data_split import stratified_split
from src.oversample import oversample_rare_classes
from src.generate_yaml import generate_yaml

BASE = Path(__file__).parent.parent

# stratified_split(
#     annotations_path=BASE / "data/train/annotations.json",
#     images_dir=BASE / "data/train/images",
#     output_dir=BASE / "data_test",
#     val_size=0.15,
#     seed=42,
# )

# oversample_rare_classes(
#     annotations_path=BASE / "data/train/annotations.json",
#     images_dir=BASE / "data/train/images",
#     output_dir=BASE / "data/train_balanced",
#     min_annotations=50,
#     seed=42,
# )

generate_yaml()

def test_balanced_exists():
    assert (BASE / "data/train_balanced/annotations.json").exists()
    with open(BASE / "data/train_balanced/annotations.json") as f:
        coco = json.load(f)
    from collections import Counter
    counts = Counter(ann["category_id"] for ann in coco["annotations"])
    under_50 = [cat_id for cat_id, cnt in counts.items() if cnt < 50]
    print(f"Images after oversample: {len(coco['images'])}")
    print(f"Classes still under 50: {len(under_50)}")
    assert len(coco["images"]) > 3905  # должно быть больше оригинала

def test_yaml_exists():
    assert (BASE / "bottles.yaml").exists()
    content = (BASE / "bottles.yaml").read_text()
    assert "nc: 369" in content
    assert "train_balanced" in content