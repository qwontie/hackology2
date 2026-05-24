#!/usr/bin/env bash
# Reproduce every weight that feeds the public-LB champion ensemble (0.8116).
#
# Usage:
#   bash train_everything.sh all              # serial: teacher -> student -> y11l -> y11x_v2 -> y8x_cb
#   bash train_everything.sh <step>           # one step (teacher, student_cwd, yolo11l_student, yolo11x_v2, yolov8x_cb, yolo11l_1536)
#   SMOKE=1 bash train_everything.sh all      # 2-epoch sanity check
#
# Parallelising across boxes (recommended — single-box serial is ~24h):
#   Box A (must run teacher first, then student CWD depends on it):
#       bash train_everything.sh teacher && bash train_everything.sh student_cwd
#   Box B:  bash train_everything.sh yolo11l_student
#   Box C:  bash train_everything.sh yolo11x_v2
#   Box D:  bash train_everything.sh yolov8x_cb
#
# Each step writes runs/detect/<name>/weights/best.pt and ultralytics dumps
# the full hyperparams to runs/detect/<name>/args.yaml — keep both for repro.

set -euo pipefail
cd "$(dirname "$0")"

STEP="${1:-all}"
SMOKE_ARG=""
[[ "${SMOKE:-0}" == "1" ]] && SMOKE_ARG="--smoke"

run_step() {
    local script="$1"
    echo
    echo "=================================================================="
    echo "  $(date -u +%H:%M:%S) — running training/$script $SMOKE_ARG"
    echo "=================================================================="
    bash "training/$script" $SMOKE_ARG
}

case "$STEP" in
    prepare)            run_step 00_prepare_data.sh ;;
    teacher)            run_step 01_teacher.sh ;;
    student_cwd)        run_step 02_student_cwd.sh ;;
    yolo11l_student)    run_step 03_yolo11l_student.sh ;;
    yolo11x_v2)         run_step 04_yolo11x_v2.sh ;;
    yolov8x_cb)         run_step 05_yolov8x_cb.sh ;;
    yolo11l_1536)       run_step 06_yolo11l_1536.sh ;;
    all)
        # Champion ensemble = 4 models. yolo11l_1536 is optional and NOT in the
        # champion config, so we skip it here. Serial order respects deps
        # (student_cwd depends on teacher; the other three are independent).
        run_step 00_prepare_data.sh
        run_step 01_teacher.sh
        run_step 02_student_cwd.sh
        run_step 03_yolo11l_student.sh
        run_step 04_yolo11x_v2.sh
        run_step 05_yolov8x_cb.sh
        echo
        echo "=================================================================="
        echo "  All 5 weights trained. Copy each runs/detect/*/weights/best.pt"
        echo "  into _weights/ and run inference with:"
        echo
        echo "    uv run predict --models student teacher yolo11l \\"
        echo "        _weights/yolo11x_v2_best.pt --model-weights 1.0 1.5 0.7 0.5 \\"
        echo "        --mode semiheavy --input public_test/images \\"
        echo "        --annotations public_test/test_images.json \\"
        echo "        --output submissions/predictions.json"
        echo "=================================================================="
        ;;
    *)
        echo "unknown step: $STEP"
        echo "valid: prepare | teacher | student_cwd | yolo11l_student | yolo11x_v2 | yolov8x_cb | yolo11l_1536 | all"
        exit 2
        ;;
esac
