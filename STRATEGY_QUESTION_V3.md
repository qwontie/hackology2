# Hackology II — Round 3 question: what's the best H200 7-hour bet?

Your V1/V2 calls were great — RT-DETR survived (now mAP50=0.21 at e27 still
climbing), yolo11l survived (plateau at 0.67), nested-TTA bug fixed (semiheavy
2-model = **5.5 min on T4**, massive headroom), L4 heavy 2-model submitted →
**we're now public #1 at 0.7947** (was #2 at 0.7304).

But runner-up **aissecobs-cc is at 0.7995** (organizer model, possibly Co-DETR
based on naming). Gap = 0.0048. We want **biggest possible jump**, not just
covering the gap.

We have **7 hours and an H200 NVL ($2/hr)** budget for a side experiment that
won't disturb our stable finals path. Need your sharpest call on what to train
on H200.

## State diff since V2

- L4 heavy (student + teacher heavy WBF) submitted → **public 0.7947**, beat
  runner-up's `kf` model (0.7571), still 0.005 behind `cc` (0.7995).
- yolo11l e26 = mAP50 **0.670** (precision 0.92), val plateau in 0.65-0.67 band
  for ~10 epochs. ETA done in ~2h. Probably not the breakthrough you hoped for.
- rtdetr e27 = mAP50 **0.208** (precision 0.26, recall 0.46). Monotonic but
  slow: e22→e27 = 0.185→0.208 = ×1.12 over 5 epochs. Won't hit your 0.55
  threshold; will end at ~0.35 by e50. Already evidence DETR family is hard to
  converge on our 369-class fine-grained data.
- T4 gen3 (semiheavy 3-model: student+teacher+yolov8l, yolov8l@0.5 weight)
  currently generating, ETA <1 min. Will submit to public for diversity probe.
- yolov8l (train1) val mAP = 0.659. Weak but adds diversity.
- Per-class diagnostic on H100 #1 val (1077 imgs):
  - student mAP50_all = 0.544 on this split (different distribution than the
    val that gave 0.90)
  - Only **3 classes** have true low AP with real val data:
    - cat 280 (Pas_WHI_BleSco_BUT_1000ml) → 7/8 hits confused with 700ml variant
    - cat 144 (ChiReg_12Yo_BUT_1000ml) → 7/8 hits confused with 700ml variant
    - cat 157 (ChiReg_12Yo_KAR_700ml) → confused with sister-brand 13Yo
  - **Pattern: model fails on 1L (1000ml) bottles** — same label, different
    bottle proportions, detector can't distinguish at its inference scale.
  - Cascade classifier expected gain: +0.3-0.5pp max (only 3 classes affected).
    You and we agree it's not the main bet.

## Concrete things on the table

Teammate dislikes DETA specifically (sees rtdetr struggle and assumes any DETR
family is high-risk). We acknowledge that risk. Options sorted by our gut:

### A) **DETA-ResNet50-Objects365-pretrained** (HF transformers)
- 365-class Objects365 pretrain → 369-class fine-tune should converge faster
  than rtdetr did from COCO start
- HF transformers integration is clean (we know the codebase from research)
- Train: 30 epochs @ imgsz=1280, batch=8 on H200 → ~4h
- Risk: DETR family showed slow convergence on this data (rtdetr). DETA's
  assignment trick (NMS + box matching) is supposed to fix this but unproven on
  fine-grained 369-class.
- Integration: ~1h, add new class wrapper in predict_v2 (like RTDETR class).

### B) **Co-DETR-Swin-Tiny on Objects365-pretrain**
- If aissecobs-cc IS Co-DETR, training one closes the architecture gap
  conceptually.
- Setup risk MUCH higher (MMDetection install, config wrangling on vast box)
- Train: 30 epochs ~5h on H200
- Integration: 2-3h (different output format, needs adapter)
- Best case: matches organizer's model. Worst case: lose 6h on setup hell.

### C) **yolo11x student with class-balanced sampling**
- Same architecture family as our student (yolo11m). Bigger backbone (110M vs 20M).
- Class-balanced sampler (`WeightedRandomSampler` with 1/sqrt(count)) +
  heavier mixup for 1L variants.
- 30 epochs @ imgsz=1536 batch=12 → ~4.5h on H200
- Integration: ZERO (drop into existing predict_v2 as another YOLO weight).
- Risk: correlated with our existing yolo11m student → WBF gain likely limited.
  +0.3-0.7pp expected.

### D) **EVA-02-Large or ConvNeXt-Large classifier cascade**
- You were skeptical in V2. Diag confirmed only 3 truly weak classes → upside
  capped at ~+0.5pp. Probably not worth the integration risk.

### E) **Multi-fold student ensemble (3 seeds yolo11m, average)**
- Train 3 yolo11m students with seeds 42/123/787, mixup higher
- 3 × ~3h on H200 = 9h (too long sequentially) OR split across H100s
- Add all 3 to WBF.
- Predictable +0.3-0.5pp from seed averaging.
- Boring but safe.

### F) **DINOv3 detection head fine-tune** (if pretrained weights exist)
- Newer than DINOv2. DINOv3-Det if it exists has strong fine-grained features.
- Unknown availability of detection-head pretrained checkpoints.
- Could be high upside or a dead end based on what's actually downloadable.

### G) **Train ENSEMBLE-AWARE student** — distill from L4 ensemble output
- Generate L4 ensemble predictions on val + train
- Fine-tune student on these as pseudo-labels with ensemble-as-teacher loss
- Could compress the ensemble into a single model (faster inference)
- Or just use as new diversified student
- Unclear net mAP impact

### H) **Heavy hyperparameter sweep of WBF on val + new probe submissions**
- 28 submissions remaining. Use them.
- Tune per-model WBF weights via val grid search, submit best.
- ~30 min, no training. +0.2-0.5pp.
- Cheap insurance regardless of H200 path.

### I) **Wildcard you might see** — anything we're missing?
- We have 7h on H200, deadline 12:00 Warsaw (~8h from now).
- Final must be `uv run predict` on T4 16GB / 30min budget.
- Semiheavy 5-model already estimated <15 min on T4 (huge headroom).
- We're public #1 by 4pp over runner-up's `kf`, behind organizer's `cc` by 0.5pp.

## Key constraints

1. Must integrate into `predict_v2.py` (its WBF ensemble) cleanly — adapter
   layer OK, but no new pipeline.
2. T4 finals: 30-min budget, semiheavy 5-model already uses ~15 min. So new
   model can use up to 10 min on T4 inference at most.
3. Public LB has ~22 subs left (we've used 8). Worth burning 3-5 on probes.
4. Private holdout domain unknown. Possible domain shift from public.
5. **Teammate concern about DETA**: bias confirmed by rtdetr slow convergence.
   We can hedge by running both DETA train AND yolo11x train in parallel (if
   we get a 2nd box for $2/hr) — but adds complexity.

## What we're asking you

1. **What's the SINGLE best H200 7-hour bet for maximizing public-LB jump?**
   Specifically rank A vs B vs C vs E vs F vs G — or recommend something we
   haven't listed. Be precise about expected mAP delta and risk.

2. **Are we missing a non-training move that would beat all training paths?**
   E.g., SAHI tile inference, model soup, EMA averaging of existing weights,
   per-class WBF weights from val grid search, etc. — given 7h, would any of
   these beat option (X) above?

3. **Teammate veto on DETA — is their fear justified?** Specifically:
   - rtdetr-l shows mAP 0.208 at e27. Does this generalize to DETA failing
     similarly, or is DETA's NMS + assignment specifically designed to fix the
     slow-convergence problem we're seeing?
   - If we DO train DETA, what specific hyperparams maximize chance of success
     given our data (369-class fine-grained alcohol bottles, mostly 700ml
     bottles, some 1L variants that look identical at detector scale)?

4. **One thing to do RIGHT NOW (next 30 min) regardless of training choice?**
   We've already done: per-class diag, viz, semiheavy smoke, P1 submit, fixed
   nested-TTA bug, fixed off-by-one in diag. What's the highest-leverage 30-min
   action while we wait for trainings + H200 decision?

5. **Risk we should worry about most right now?**

Files you may want to look at:
- `predict_v2.py` (inference pipeline, mode definitions)
- `scripts/per_class_ap.py` + `diagnostics/per_class_student.json` (diag)
- `data_probe/STATE.md` (current state)
- `STRATEGY_QUESTION_V2.md` (round 2 context)
