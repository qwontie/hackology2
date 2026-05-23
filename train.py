from src.train_models import *

from pathlib import Path
from src.convert_coco_yolo import coco_to_yolo

match input("Enter mode: t1, t2, or t3"):
    case "t1":
        train1()
    case "t2":
        train2()
    case "t3":
        train3()
