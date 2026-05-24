# Reproducing the final ensemble

This directory reproduces every weight that feeds the final WBF ensemble in
`predict_v2.py`. Each numbered script is one independent training run; pass
`--smoke` to any script for a 2-epoch sanity check.

The actual hyperparameters used by every run we shipped are checked in under
`training_metadata/<box>/<run>_args.yaml` (ultralytics auto-dumps these next
to every `best.pt`). The launchers below mirror those args.

## Dependency graph

```
00_prepare_data.sh                   (one-time, makes data/{train,val}/{images,labels})
        │
        ├──► 01_teacher.sh           (yolo11x @ 1536, 40 ep)              ──► teacher_x_1536_all_best.pt
        │           │
        │           └──► 02_student_cwd.sh   (yolo11m + CWD distill)     ──► student_m_1536_cwd_best.pt
        │
        ├──► 03_yolo11l_student.sh   (yolo11l @ 1536, 45 ep, fresh)      ──► yolo11l_student_best.pt
        │
        ├──► 04_yolo11x_v2.sh        (yolo11x @ 1280, 30 ep, seed=42)    ──► yolo11x_v2_best.pt   ⭐ +0.0074 LB
        │
        ├──► 05_yolov8x_cb.sh        (yolov8x class-balanced, 30 ep)     ──► yolov8x_cb_best.pt
        │
        └──► 06_yolo11l_1536.sh      (yolo11l @ 1536 scale-variant)      ──► yolo11l_1536_best.pt  (optional)
```

## Models that fed the public-LB champion (0.8116)

| weight | role | LB contribution |
|---|---|---|
| `student_m_1536_cwd_best.pt` | anchor (small, fast, CWD-distilled) | baseline |
| `teacher_x_1536_all_best.pt`  | anchor (large, slow, accurate) | baseline |
| `yolo11l_student_best.pt`     | diversity (different arch family)  | **+0.017** vs 2-model |
| `yolo11x_v2_best.pt`          | diversity (fresh seed yolo11x)     | **+0.0074** vs 3-model |

Final WBF weights: `[1.0, 1.5, 0.7, 0.5]`, default `iou_thr=0.55`, `skip_box_thr=0.001`.

## Running everything

```bash
# Train every model that ships in the champion ensemble (serial, ~12-14h on a single H100):
bash train_everything.sh all

# Train a single model:
bash train_everything.sh teacher
bash train_everything.sh student_cwd
bash train_everything.sh yolo11l_student
bash train_everything.sh yolo11x_v2
bash train_everything.sh yolov8x_cb

# Parallelise across boxes — give each box one model:
#  Box A:  bash train_everything.sh teacher && bash train_everything.sh student_cwd
#  Box B:  bash train_everything.sh yolo11l_student
#  Box C:  bash train_everything.sh yolo11x_v2
#  Box D:  bash train_everything.sh yolov8x_cb
```

Each launcher writes `runs/detect/<name>/weights/best.pt`. After all models
finish, copy each `best.pt` into `_weights/` (or upload to the GH release and
let `predict_v2.py` download them via `WEIGHT_ALIASES`).

## Hardware footprint actually used

| model | GPU | wall time | batch | imgsz | epochs |
|---|---|---|---|---|---|
| teacher_x_1536_all     | H100 80GB | ~8h  | 12 | 1536 | 40 |
| student_m_1536_cwd     | H100 80GB | ~6h  | 8  | 1536 | 35 |
| yolo11l_student        | H100 80GB | ~3h  | 16 | 1280 | 30 |
| yolo11x_v2             | H100 80GB | ~3h  | 14 | 1280 | 30 |
| yolov8x_cb             | H200 NVL  | ~3h  | 12 | 1536 | 30 |
| yolo11l_1536 (optional)| H100 80GB | ~5h  | 12 | 1536 | 25 |

A single 24GB consumer card (3090/4090) can run any of these at `batch=4–6`
and `imgsz=1280` with roughly 2-3× longer wall time.

## Inference reproduction (T4 16GB / 30 min budget)

After all weights are in place:
```bash
uv run predict \
    --models student teacher yolo11l _weights/yolo11x_v2_best.pt \
    --model-weights 1.0 1.5 0.7 0.5 \
    --mode semiheavy \
    --input public_test/images \
    --annotations public_test/test_images.json \
    --output submissions/predictions.json
```

`semiheavy` mode = 2 scales (1280, 1536), no ultralytics TTA, no flip. The
exact run takes ~13 min on a Colab T4 (well under the 30 min eval budget).
