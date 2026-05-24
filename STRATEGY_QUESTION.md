# Hackology II — Strategic question for second-opinion AI

We're 10h from deadline of an alcohol-bottle object-detection hackathon (369 SKU
classes, metric **mAP@0.5**, COCOeval). We currently lead the public leaderboard
at **0.7304** vs runner-up at 0.6926. Final evaluation runs our `predict_v2.py`
on a **private holdout** in a fixed environment (**Tesla T4 16 GB, 30 min wall
budget**). Public-test images are sampled from one source (`DPR_MIR_3963`); the
private holdout's domain is unknown to us.

We have abundant compute: a rented T4 box that mirrors the finals env, plus two
H100 80 GB boxes. Read **`data_probe/STATE.md`** for the full project snapshot
(SSH creds, paths, what's running, weight URLs). Read **`predict_v2.py`** and
**`src/train_models.py`** for the inference + training entry points.

## What works right now

Profile on the T4 mirror (real `predict_v2.py` inference on 481 public-test
images, 30 min hard budget, 25 min safety target):

| Test | Models | Mode (passes / image) | Time   | Verdict           |
|------|--------|----------------------|--------|-------------------|
| L1   | student                    | balanced (1536 + flip = 2)        | 78 s   | ✅ fits           |
| L2   | student                    | heavy (3 scales × 2 flips = 6)    | 247 s  | ✅ fits           |
| L3   | student + teacher          | balanced (2 models × 2 = 4)       | 248 s  | ✅ fits           |
| **L4** | student + teacher        | **heavy (2 models × 6 = 12)**     | **833 s ≈ 14 min** | ✅ **current champion** |
| L5   | student + teacher + teacher_dprft | heavy (3 × 6 = 18)        | ~34 min projected (killed early) | ❌ **over 30 min** |

Headline: **`heavy` mode with ≥3 models does not fit on the T4 finals box.**
So additional models must run in `balanced` mode in the ensemble.

## What our weights are

In GitHub Release `qwontie/hackology2 v0-weights`:

| weight | size | what |
|---|---|---|
| `student_m_1536_cwd_best.pt` | 39 MB | yolo11m, CWD-distilled from teacher, val mAP50 **0.9002**, public 0.7304 |
| `teacher_x_1536_all_best.pt` | 110 MB | yolo11x teacher Phase 1 (real + DPR_MIR + synth), val mAP50 **0.9253** |
| `teacher_x_1536_dprft_best.pt` | 110 MB | yolo11x Phase 2 fine-tuned on DPR_MIR subset, val mAP50 **≈ 0.91** |

Plus, **in training right now**:
- H100 #2: `yolov8l` (teammates' train1), full data, 50 ep @ 1280, batch 32, val mAP50 reached **0.65 by epoch 39/50**, finishing in ~5 min.
- H100 #1: `rtdetr-l` (teammates' train3, **just relaunched after a collapse — see below**), 50 ep @ 1280, batch 15, ETA ~2.6 h.

## Training data composition (important)

`data/train_balanced/annotations.json` contains **all 3 sources**:

| source | images | notes |
|---|---|---|
| DPR_MIR_3963_QUALITY_EVALUATION_09072025 | 1369 | same source as public_test |
| SIDG_TRAIN | 1069 | other real images |
| SIDG_SYNTH_TRAIN | 954 | **synthetic** (~28% of train volume) |
| total | 3392 imgs, 142 682 anns | |

So synthetic is already in the student/teacher training. The val split also
contains all three sources. Private holdout source is unknown.

## What just broke / what's risky

**RT-DETR collapse on first attempt.** Teammates' `train3()` used `YOLO("rtdetr-l.pt")`
with YOLO-default hyperparams: lr0=0.01, AMP on, heavy aug (mosaic=0.5,
degrees=15, shear=5). Result over 4 epochs:

| epoch | mAP50 | recall | comment |
|---|---|---|---|
| 1 | 4.37e-07 | 0.00142 | warm |
| 2 | 4.74e-07 | 0.00151 | flat |
| 3 | 1.75e-07 | 0.000688 | regression |
| 4 | 6.83e-08 | 0.000288 | collapsing |
| 5 | 1.58e-06 | 0.000368 | tiny rebound but still 10⁶× too small |

Reads as classifier-head collapse (cls_loss settled at ~0.07; for 369 classes
the floor should be near `ln(369) ≈ 5.9`).

We've just relaunched with DETR conventions: `RTDETR(...)` class,
`optimizer="AdamW"`, `lr0=1e-4`, `weight_decay=1e-4`, `warmup_epochs=2`,
`amp=False`, no mosaic/mixup, no degrees/shear, 50 epochs. Whether this trains
to anything useful in ~2.5 h is unknown.

## What we're considering for the next 10 hours

### Stays as-is

1. T4 box keeps profiling ensembles to confirm timings as new weights arrive.
2. H100 #1 trains rtdetr-l v2 (~2.5 h, gambling that DETR-friendly hyperparams converge).
3. H100 #2 finishes train1 (yolov8l) in ~5 min.

### Open: what to run on H100 #2 after train1 finishes

We have ~9 h on that box. Options, sorted roughly by our current intuition:

| option | ETA | what it gives | downside |
|---|---|---|---|
| **a) Pseudo-label distillation of student** | ~1 h | Generate pseudo-labels on public_test using the L4 ensemble (`conf ≥ 0.65`), then fine-tune `student_m_1536_cwd_best.pt` for 3-4 epochs at lr 1e-4 on real+pseudo. Hoped to nudge student toward public-domain visuals (and, if private ≈ public, to private too). | Public ≡ DPR_MIR which is already in train; expected gain small (~0.3-0.5 %). Risk of fitting noise. |
| **b) Train a yolo11l student from scratch on full data** | ~3-4 h | Bigger model (val mAP probably 0.91-0.93). Adds size diversity to WBF. | Tight against deadline; if it doesn't finish, time wasted. Same arch family as yolo11m student. |
| **c) Train yolo11m student with different seed + heavier aug** | ~1.5 h | Cheap "model soup" ingredient — averaging two student checkpoints or WBF-fusing both often gives +0.3-0.5 % free. | Marginal gain; same arch as our current student. |
| **d) Re-train teacher / student WITHOUT synth** | ~2-3 h | Tests the hypothesis that synth hurts on the real-image private holdout. If it wins on val, it's the new student. | Speculative; flips our long-held assumption. |
| **e) Train MGD-distilled student** (yolo-distill-mgd is in `vast_snapshot/`) | ~2-3 h | Alternative distillation loss; produces a student with different error modes — pure WBF diversity. | Newer fork, less battle-tested; uncertain integration time. |

### Open: how to compose the final 5-model ensemble

Target stack (if rtdetr-l survives):
`student_m + teacher_x + teacher_dprft_x + yolov8l + rtdetr-l`,
all `balanced` (1536 + flip), WBF-fused, **+ teammates' temperature
calibration on val** (already implemented in `src/calibration.py`).

Estimated timing: 5 models × 2 passes = 10 passes, ≈ 12-15 min — comfortably
under budget. But this is **extrapolation**; we haven't actually run a 5-model
ensemble yet, and L5 (3 models heavy) slowed unexpectedly on T4 (0.37 → 0.23
img/s by the time we killed it).

If rtdetr-l doesn't converge: 4-model ensemble (drop rtdetr-l). If yolov8l also
turns out weak (val mAP 0.65 is modest), we'd weight it lower in WBF or drop it
and stay with the original 3 models (student + 2 teachers, all balanced).

## What we're asking you

1. **H100 #2 best use** — given the constraint that we cannot run heavy mode
   with >2 models, which option (a-e, or something we missed) maximises expected
   private-leaderboard improvement? We have only one model-train slot on H100 #2
   (~5-9 h depending on choice). Our gut says (a) pseudo-label is fastest,
   (b) yolo11l student is highest-upside but risky on timing.

2. **rtdetr-l strategy** — is it worth the GPU-hours? If DETR-friendly hyperparams
   don't show clear mAP climb by epoch 10 (~25 min), should we kill it and use
   that GPU for something else? What would you do?

3. **WBF composition** — for 5 models in WBF, would you set unit weights, or
   weight by val mAP? Should we drop `teacher_dprft` (it's a fine-tune of
   `teacher_all` so they're heavily correlated)?

4. **Anything obviously missing** — given the situation (public 0.7304, ~10 h
   left, T4 30 min budget, unknown private distribution), what high-value move
   are we *not* considering? Especially around:
   - Test-time tricks that don't need re-training (SAHI for small objects?
     Class-specific confidence thresholds from val PR curves?).
   - Robustness moves in case private has a domain shift (lower-res images,
     occluded bottles, different lighting, new ratios).
   - Submission strategy with our remaining 28 of 30 public submissions.

5. **Risk we should be most worried about** — what's the most likely way our
   plan loses to the runners-up between now and deadline?

Please be concrete (specific file paths, parameters, commands). We have
the engineering bandwidth to act on whatever you recommend — just need the
sharpest call.
