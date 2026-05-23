from pathlib import Path

from src.data_split import stratified_split
from src.oversample import oversample_rare_classes
from src.generate_yaml import generate_yaml

BASE = Path(__file__).parent.parent

stratified_split(
    annotations_path=BASE / "data/train/annotations.json",
    images_dir=BASE / "data/train/images",
    output_dir=BASE / "data_test",
    val_size=0.15,
    seed=42,
)