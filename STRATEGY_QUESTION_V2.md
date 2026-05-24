# Hackology II — Round 2 strategy question for second-opinion AI

You reviewed our plan ~3h ago (`STRATEGY_QUESTION.md`). You correctly killed
pseudo-label distillation and temperature calibration, validated the yolo11l
student bet, and called out the critical WBF-weights / entry-point bugs we then
fixed. We need round 2.

## State diff since you last looked (00:00 → 03:00 Warsaw)

### Leaderboard moved
- **SyntaxDebuggingDivas (team-15) is now #1 at 0.7552**, we dropped to #2 at 0.7304.
- ~9h until deadline (2026-05-24 12:00 Warsaw).

### Trainings status
- **yolo11l student** (your rec, H100 #2): epoch 11/45, val mAP50 plateauing at
  **0.62-0.63** since e8. Slower than we hoped — recipe was lr0=1e-3, AdamW, cos_lr,
  mosaic=1.0 mixup=0.05 copy_paste=0.2 close_mosaic=10, imgsz=1536 batch=16. Hopes
  for second wind after close_mosaic at e35; ETA ~3h.
- **rtdetr-l v2** (H100 #1): just hit e15 mAP50 = **0.118**. Trajectory has been
  monotonic × 1.3-1.9 per epoch since e8. Will plausibly cross your kill threshold
  (0.15 at e10-12) at e16-17. Decision: keep alive, ETA ~2h to e30.
- **yolov8l (teammates' train1)**: finished at val mAP50 **0.659**. Uploaded to GH
  Release. Will join WBF with low weight (0.5) for arch diversity.

### Smoke test issue — IMPORTANT
We launched `time uv run predict --input ... --output /tmp/smoke.json` on T4
(real finals env, default env = heavy mode + student+teacher) ~30 min ago. **It's
STILL running. GPU 100% util, 1.6GB memory used (single model loaded).** L4
profile on T4 said 833s for the same config. We're at 30+ min and not done.

This suggests either (a) the L4 833s figure was wrong / on a warmer/faster T4,
or (b) cold-start kernel compile is huge, or (c) public_test images differ from
what we profiled. If finals env behaves the same → **we will time out**. We need
to debug AND consider a faster fallback config.

### Per-class diagnostic running NOW
Running `scripts/per_class_ap.py` on H100 #1 (parallel with rtdetr training).
Computing per-class AP50 on val for student (val=1077 imgs on H100 #1's data
slice — apparently different from H100 #2's val=582; possibly H100 #1 has
combined-source val, H100 #2 has DPR_MIR-only — we'll verify). Will identify
top-30 weakest classes plus their worst val images for human review and
targeted action.

## What we're considering for the last 9h (post-diagnostic)

We're seriously thinking of **buying an H200 NVL** ($2/hr from a saved
vendor) to add a 4th training slot. Tell us if you think this is overkill or
not.

### Wild ideas, sorted by our gut on expected gain

#### A) **Detector+Classifier cascade** ← we think this is the biggest unexplored lever
The fundamental bottleneck is fine-grained 369-class SKU classification, not
localization. At imgsz=1536 a single bottle is maybe 100-300px; a dedicated
classifier on cropped bottles sees 384-512px of label.

Plan:
- Generate train crops from `train_balanced/annotations.json` using GT bboxes
  (~142k crops). Resize to 384, train **EVA-02-Base** or **ConvNeXt-Base** on
  369-way classification, 30 epochs ~3h on H200.
- Inference cascade in new `predict_v3.py`: ensemble produces detections →
  for each det, crop+resize → classifier produces 369-way distribution → fuse:
  `final_score[c] = det_score * classifier_prob[c]^α` and optionally
  reassign class if classifier is much more confident on a different label.

Budget check: ~100 dets/image × 481 imgs × ~10ms/crop on T4 = ~8 min. With L4
ensemble at 14 min, total = 22 min. **Fits in 30-min budget.**

Risks:
- New pipeline; integration time ~3-4h (including val verification).
- If classifier's class assignment is too confident in wrong direction, can
  *worsen* mAP — needs val sweep of α and reassign-threshold.
- Cropping detector boxes at unknown scale: padding / aspect-preserving resize
  needs to match training.

Expected: +1-3pp on mAP. Closes the gap to leader and gives runway.

#### B) **yolo11x student v2 with class-balanced sampling**
Bigger backbone than current yolo11m student, plus class-aware sampling to fix
under-represented classes (which our diagnostic will identify).
- 30-40 epochs at imgsz=1536, 3-4h on H200.
- Distill from teacher (CWD or just hard labels with TTA).
- Add to ensemble.
- Expected: +0.5-1.5pp.
- Risk: low — proven architecture, just bigger.

#### C) **SAHI for one ensemble pass** (cheap)
Add a single tile-inference pass (2×2 with 25% overlap) on student to better
catch small/occluded bottles in crowded scenes.
- ~5 min extra on T4 budget — tight but fits.
- Expected: +0.3-1pp on crowded images.
- Risk: low; if it doesn't help on val, just don't include.

#### D) **Per-class data augmentation / oversampling retrain** (after diagnostic)
Once we know which classes are AP=0 or AP<0.2 on val, retrain student with:
- 5-10× oversampling of weak-class samples.
- Heavier aug on those crops (color, paste-into-bg).
- Reuse our existing student weights as warm start (10 epoch fine-tune).
- 1.5-2h on H100/H200.
- Expected: dependent on diagnostic — could be 0pp (if weak classes are also
  weak on private) or +1-2pp (if simply data scarcity).

#### E) **VLM zero-shot re-rank** for uncertain detections
For dets with score in [0.3, 0.6], send crop + class names list to a strong
VLM (CLIP-G, SigLIP-L, or DINOv2 with class-name embeddings via category names)
and use VLM logit as tie-breaker for class assignment.
- No re-training needed.
- Budget: VLM inference cost on T4 — small models OK (~100MB SigLIP-S).
- Expected: +0.2-0.8pp.
- Risk: integration time; VLM might not know specific alcohol brand designs.

## Submission probe plan (we have 28 of 30 left)

We're submitting L4 baseline (student + teacher heavy WBF) RIGHT NOW as P1 — we
have it cached as `submissions/L4_ensemble_heavy.json`. This will confirm whether
the ensemble actually beats single-student 0.7304 AND tells us the floor.

After P1, plan:
- P2: L4 + yolov8l (4-model balanced — once we re-profile heavy timing)
- P3: 5-model balanced with yolo11l once trained
- P4-P7: weight tuning + SAHI / cascade probes
- Final: best config wins

## What we're asking you

1. **Cascade go/no-go?** Is detector+classifier cascade the right unexplored
   lever, given:
   - 369 fine-grained SKU classes (clearly classification-limited)
   - 3h H200 training budget for ConvNeXt-Base or EVA-02-Base
   - 8 min inference cost in T4 budget (we have ~16 min headroom from L4)
   - Need new code path in predict_v3.py
   
   Or do you see a cheaper win we're missing?

2. **If yes to cascade — which classifier?** EVA-02-Base (88M, ~6ms/crop on T4)
   vs ConvNeXt-Base (89M, ~5ms/crop) vs SigLIP-Base-384 (203M, ~10ms/crop with
   text encoder for class names — gives zero-shot fallback). We're leaning
   EVA-02-Base for raw classification quality on ImageNet-21k fine-grained.

3. **H100 #2 strategy after yolo11l finishes (~3h)** — second train slot opens.
   Options: (a) yolo11x v2 with class-balanced sampling (3-4h, just makes
   deadline), (b) help cascade classifier training on H200 + yolo11x train on
   H100 (parallel), (c) re-train student WITHOUT synth (test if synth hurts on
   private), (d) something else.

4. **Smoke test 30+ min — what's the likely cause?** L4 profile was 833s for
   same heavy student+teacher on T4. Now 30+min and counting. Should we (a)
   drop to balanced mode (~250s, gives up TTA), (b) try heavy on fewer scales
   (drop 1920 → keep 1280+1536), (c) drop to single-model heavy?

5. **Weight grid search via submissions** — we have 28 subs. Worth burning 5-10
   on WBF-weight optimization for the final ensemble, or is the prior weak?

6. **Risk we should worry about most** with this revised plan?

We have engineering bandwidth to act on whatever you call. We just need the
sharpest take in the next 30-60 min so we can commit resources before deadline
pressure hits.

Files updated since V1:
- `predict_v2.py` (env defaults, WBF weights propagation, RTDETR class)
- `pyproject.toml` (predict entry point fixed)
- `src/train_models.py` (rtdetr DETR-friendly hyperparams)
- `scripts/train_yolo11l.py` (your recipe)
- `scripts/per_class_ap.py` (diagnostic, running now on H100 #1)
- `scripts/viz_weak_classes.py` (visualization after diagnostic)
- `data_probe/STATE.md` (live state, SSH coords, monitors)
