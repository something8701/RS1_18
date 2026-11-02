#!/usr/bin/env bash
# evidence.sh: [yolo|drone|all]: gather the prepared evidence for the testing part into eval/demo/evidence/ (the …
source "$(dirname "$0")/demo_env.sh"
Y=$WS/eval/yolo
OUT=$DEMO_DIR/evidence
mkdir -p "$OUT"
grab() {
  [ -f "$1" ] || { warn "missing $1"; return; }
  cp "$1" "$OUT/$2"; ok "$2"
  [ -z "$NO_BROWSER" ] && { (xdg-open "$OUT/$2" > /dev/null 2>&1 &); sleep 0.3; }
}
case ${1:-all} in yolo|all)
  say "YOLO detector"
  grab "$Y/label_check.jpg"                                         yolo_1_auto_labels.jpg
  grab "$Y/runs/evidence_v1/results.png"                            yolo_2_training_curves.png
  grab "$Y/runs/evidence_v1/confusion_matrix_normalized.png"        yolo_3_confusion_val.png
  grab "$Y/runs/evidence_v1_test/confusion_matrix_normalized.png"   yolo_4_confusion_unseen_world.png
  grab "$Y/runs/evidence_v1_test/val_batch0_pred.jpg"               yolo_5_unseen_world_predictions.jpg
  [ -f "$Y/test_result.txt" ] && { cp "$Y/test_result.txt" "$OUT/yolo_test_result.txt"; cat "$Y/test_result.txt"; } ;;
esac
case ${1:-all} in drone|all)
  say "Drone tree model"
  grab "$SRC/test_evidence/dense_forest_model_progress.png"     drone_1_model_progress.png
  grab "$SRC/test_evidence/dense_forest_new_model_comparison.png" drone_2_comparison.png
  for g in gate1_sparse_no_change_final_2026-09-30.txt gate3_dense_two_pass_no_change_final_2026-09-30.txt; do
    grep -m1 -E "PASS|FAIL" "$SRC/test_evidence/$g"
  done | tee "$OUT/drone_gates.txt" ;;
esac
say "Folder: $OUT"
