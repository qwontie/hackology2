# Hackology II — Round 4: rtdetr bug + H100 #1 use + confidence pattern

V3 → execution. Things changed:

## State diff since V3

**Submissions:**
- P1 L4 heavy = 0.7947 (won't survive T4 30-min budget — heavy = 35 min)
- P2 semiheavy 2-model = **0.7871** (real deployable ceiling)
- P3 semiheavy 2-model + sibling-copy = 0.7869 (-0.0002, **sibling-copy DEAD on public**)
- Gap to organizer's `aissecobs-cc` = 0.7995 → -0.012 to close from real deployable

**Training (now):**
- **H200 yolo11x_cb** (your V3 pick): e4 = mAP50 **0.582** (e1 0.497 → e2 0.553 → e3 0.566 → e4 0.582). Strong climb. ETA finish ~3h.
- **H100 #2 yolov8x_cb**: just started (class-balanced + sibling boost, AdamW, cos_lr). ETA ~2h. Spec: arch diversity to H200's yolo11x_cb.
- **H100 #1**: **rtdetr training crashed at e40** (mAP50 0.567 at e39 saved as best). Resume hits `'RTDETRDecoder' object has no attribute 'stride'` via YOLO CLI; switching to `RTDETR` python class hits a *separate* numpy/inspect issue (details in §A).

**Weights now in arsenal** (all uploaded to GH Release):
- `student_m_1536_cwd_best.pt` (~0.90 val)
- `teacher_x_1536_all_best.pt` (~0.90 val)
- `teacher_x_1536_dprft_best.pt` (~0.90 val)
- `yolo11l_student_best.pt` (mAP50 0.705 val)
- `yolov8l_train1_best.pt` (mAP50 0.659 val)
- `rtdetr_l_best.pt` (mAP50 0.567 val) ← rescued from crashed train
- *training:* `yolo11x_cb` (H200, climbing fast)
- *training:* `yolov8x_cb` (H100 #2, just started)

**Open submissions on T4** ready/pending:
- `predictions.json` (= L4 heavy, submitted)
- `semiheavy_2model_student_teacher.json` (P2 ref)
- `semiheavy_2model_sibling.json` (P3, sibling dead)
- `t4_semiheavy_3model.json` (+yolov8l@0.5, 3-model semi)
- `t4_semiheavy_3model_sibling.json` (3-model + sibling)
- `semiheavy_3model_y11l.json` (student+teacher+yolo11l@0.7) — **not yet submitted**, awaiting LB slot

## A. rtdetr bug — your fresh eyes welcome

H100 #1: `/workspace/v1/hackology2/.venv` (Python 3.11, ultralytics with RTDETR). The original train (model33/34) ran via `yolo detect train model=rtdetr-l.pt data=bottles.yaml ...` for **40 epochs successfully** in this same venv. Then crashed (no log of crash itself; just the process died and tmux session terminated).

**Attempt 1**: resume via `yolo detect train model=runs/model34/weights/last.pt resume=True` inside same tmux

```
File ".../ultralytics/utils/loss.py", line 168, in __init__
    self.stride = m.stride
AttributeError: 'RTDETRDecoder' object has no attribute 'stride'
```

ultralytics tries to attach v8 detection loss to an RTDETR model after resume, querying `.stride` which RTDETRDecoder doesn't expose.

**Attempt 2**: fresh train from best.pt via RTDETR class (Python):
```python
from ultralytics import RTDETR
m = RTDETR('runs/model34/weights/best.pt')
m.train(data='bottles.yaml', epochs=20, imgsz=1280, batch=15, device=0, ...)
```

In **direct ssh** `.venv/bin/python -c "from ultralytics import RTDETR; print('OK')"` → **WORKS** (prints OK).

In **tmux** (`tmux new-session -d -s X '.venv/bin/python /tmp/script.py >log 2>&1'`) → **FAILS**:
```
File ".../numpy/ma/core.py", line 7940, in <module>
    inner.__doc__ = doc_note(np.inner.__doc__, ...)
File ".../numpy/ma/core.py", line 125, in doc_note
    notesplit = re.split(..., inspect.cleandoc(initialdoc))
AttributeError: module 'inspect' has no attribute 'cleandoc'
```

Direct python in same dir has `inspect.cleandoc` present, `inspect.__file__` = `/.uv/python_install/cpython-3.11.15-linux-x86_64-gnu/lib/python3.11/inspect.py`. Same path on disk.

We tried: `source .venv/bin/activate`, `.venv/bin/python` explicit, `PYTHONNOUSERSITE=1`, removing `tee` from pipe. All fail in tmux, all work direct.

**Question 1**: known bug or env quirk we can patch in 5 min? If "no easy fix in 30 min", we're moving H100 #1 to something else.

## B. H100 #1 strategic use — "too many YOLOs"

We have 4 hours on H100 #1 with no rtdetr if you can't unstick it.

Currently planned ensemble (5 models): student + teacher + yolo11l + yolov8l@0.5 + (yolo11x_cb OR yolov8x_cb when ready).

Adding ANOTHER YOLO family member feels like correlated diversity → low WBF gain. We're tempted to just **shut H100 #1 down** ($1.5/hr × 4h = $6 savings) rather than spend on yet another correlated weight.

**Question 2**: Best H100 #1 use given current arsenal?

Options:
- **B1**: yolo11x fresh different seed (vanilla, no class-balance, seed=42) — adds basin diversity to H200's yolo11x_cb, +0.1-0.3pp expected. **The "boring" pick from V3 option E.**
- **B2**: yolo11l fresh different seed + DPR-only data — quick 1.5h, low yield
- **B3**: yolo11x at imgsz=1920 (larger inference scale, captures fine SKU details) — slower train ~5h, may not finish in time
- **B4**: rtdetr-x larger (if numpy bug fixable) — could finally crack the >0.7 barrier with transformer diversity
- **B5**: Shut down, save money — accept 5-model ensemble with H200+H100#2 outputs
- **B6**: Use H100 #1 for **inference batching** (run multiple WBF weight combos on val to grid-search optimal weights — 10-20× faster than T4 grid)
- **B7**: Your wildcard

We lean **B6** or **B5**. **B1** is "spend time on more of the same."

## C. Confidence pattern question

Teammate keeps pushing **confidence calibration** because:
- 39 classes where model never predicts at conf > 0.5 in val (27 have non-empty val GT)
- Average prediction confidence across all classes is ~0.10-0.25 at conf=0.001 threshold (heavily right-skewed; few preds above 0.7)
- "Many classes have low confidence" — felt intuitively wrong by us

Our take (please correct):
- mAP@0.5 is computed per-class as **AP integral over PR curve** at IoU=0.5 — it's **threshold-invariant** within class.
- Per-class hard threshold cutoff (drop preds < X) won't change AP because AP already integrates over all thresholds.
- Per-class score rescaling (multiply by K) preserves rank order → no AP change.
- So calibration is **null op** for mAP.
- Real gain pathways: better rank order (ensemble WBF reranks) or recover missed TPs (sibling-copy attempted, public dead).
- **Conclusion**: this is a fine-grained 369-class artifact, not a fixable problem — low confidence ≠ low mAP.

**Question 3**: Confirm or correct. If something specific can squeeze 0.001-0.003 from confidence handling, what's the exact mechanism?

## D. Gemini vision review — incoming

We're about to feed Gemini 20 images at conf≥0.3 (clean readable preds) + GT overlay + categories. Asking Gemini specifically: "what does the model get systematically wrong that we'd never catch from numbers alone?" Will share Gemini's report next round.

**Question 4**: What specifically should we ask Gemini? Sample questions we have:
1. "What confusion patterns do you see?" (sibling-style? domain-style?)
2. "Are there annotation errors in GT?"
3. "What labels look hard for any model to get right vs. just hard for ours?"
4. "Geometric mistakes (boxes too small/large)?"

Anything sharper?

## E. Time map

- Now: ~04:10 Warsaw, deadline 12:00 → ~7h 50m left.
- H200 yolo11x_cb ETA: ~3h remaining.
- H100 #2 yolov8x_cb ETA: ~2h.
- T4 final smoke: 0.5h.
- Public LB submissions left: ~22.

## Files for you

- `predict_v2.py` — inference pipeline + `--sibling-copy` opt-in
- `scripts/sibling_copy.py` — postprocess we wrote based on V3 (works locally, dies on public)
- `scripts/build_balanced_trainlist.py` + `scripts/launch_yolo11x_cb_h200.sh` — H200 training recipe
- `STRATEGY_QUESTION_V3.md` — last round
- `data_probe/STATE.md` — state snapshot
- `submissions/` — all candidate JSONs
- `diagnostics/per_class_student.json` — per-class AP report
- `diagnostics/viz_specific/SUMMARY.json` — confusion patterns 280/144/157

Punchy answers preferred. Esp. on Q1 (rtdetr unstick or move on) and Q2 (H100 #1 use).
