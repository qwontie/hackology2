import json
import tempfile
from pathlib import Path

from src.ensemble import wbf_ensemble, load_predictions, load_image_info

BASE = Path(__file__).parent.parent

# Фейковые данные для тестов
FAKE_TEST_IMAGES = {
    "images": [
        {"id": 1, "file_name": "img1.jpg", "width": 1000, "height": 1000},
        {"id": 2, "file_name": "img2.jpg", "width": 1000, "height": 1000},
    ]
}

FAKE_PREDS_MODEL1 = [
    {"image_id": 1, "category_id": 5, "bbox": [100, 100, 200, 200], "score": 0.9},
    {"image_id": 1, "category_id": 5, "bbox": [105, 105, 195, 195], "score": 0.85},  # почти то же что model2
    {"image_id": 2, "category_id": 10, "bbox": [300, 300, 100, 100], "score": 0.7},
]

FAKE_PREDS_MODEL2 = [
    {"image_id": 1, "category_id": 5, "bbox": [102, 102, 198, 198], "score": 0.88},  # перекрывается с model1
    {"image_id": 2, "category_id": 10, "bbox": [305, 305, 95, 95], "score": 0.75},
]


def make_tmp_json(data: dict | list, tmp_dir: str) -> Path:
    p = Path(tmp_dir) / f"tmp_{id(data)}.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_load_image_info():
    with tempfile.TemporaryDirectory() as tmp:
        p = make_tmp_json(FAKE_TEST_IMAGES, tmp)
        info = load_image_info(p)
        assert info[1] == (1000, 1000)
        assert info[2] == (1000, 1000)


def test_load_predictions():
    with tempfile.TemporaryDirectory() as tmp:
        p = make_tmp_json(FAKE_PREDS_MODEL1, tmp)
        preds = load_predictions(p)
        assert len(preds) == 3
        assert preds[0]["category_id"] == 5


def test_wbf_merges_overlapping_boxes():
    with tempfile.TemporaryDirectory() as tmp:
        p1 = make_tmp_json(FAKE_PREDS_MODEL1, tmp)
        p2 = make_tmp_json(FAKE_PREDS_MODEL2, tmp)
        ti = make_tmp_json(FAKE_TEST_IMAGES, tmp)

        result = wbf_ensemble([p1, p2], test_images_path=ti, iou_thr=0.5)

        img1_preds = [r for r in result if r["image_id"] == 1]
        img2_preds = [r for r in result if r["image_id"] == 2]

        # перекрывающиеся боксы должны слиться
        assert len(img1_preds) < 3, f"Expected merged boxes, got {len(img1_preds)}"
        assert len(img2_preds) == 1

        print(f"img1 boxes after WBF: {len(img1_preds)} (было 3)")
        print(f"img2 boxes after WBF: {len(img2_preds)}")


def test_wbf_output_format():
    with tempfile.TemporaryDirectory() as tmp:
        p1 = make_tmp_json(FAKE_PREDS_MODEL1, tmp)
        ti = make_tmp_json(FAKE_TEST_IMAGES, tmp)

        result = wbf_ensemble([p1], test_images_path=ti)

        assert len(result) > 0
        for pred in result:
            assert "image_id" in pred
            assert "category_id" in pred
            assert "bbox" in pred
            assert "score" in pred
            assert len(pred["bbox"]) == 4
            assert 0.0 < pred["score"] <= 1.0


def test_wbf_equal_weights_vs_custom():
    with tempfile.TemporaryDirectory() as tmp:
        p1 = make_tmp_json(FAKE_PREDS_MODEL1, tmp)
        p2 = make_tmp_json(FAKE_PREDS_MODEL2, tmp)
        ti = make_tmp_json(FAKE_TEST_IMAGES, tmp)

        result_equal = wbf_ensemble([p1, p2], test_images_path=ti, weights=[1.0, 1.0])
        result_custom = wbf_ensemble([p1, p2], test_images_path=ti, weights=[2.0, 1.0])

        # scores должны отличаться при разных весах
        scores_equal = sorted([r["score"] for r in result_equal])
        scores_custom = sorted([r["score"] for r in result_custom])
        assert scores_equal != scores_custom, "Weights should affect scores"

        print(f"Equal weights scores:  {scores_equal}")
        print(f"Custom weights scores: {scores_custom}")