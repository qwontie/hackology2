from pathlib import Path

from src.data_split import stratified_split
from src.oversample import oversample_rare_classes
from src.generate_yaml import generate_yaml

stratified_split(
    annotations_path=Path("../data/train/annotations.json"),
    images_dir=Path("../data/train/images"),
    output_dir=Path("./data_test"),
    val_size=0.15,
    seed=42,
)