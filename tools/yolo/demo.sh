#!/usr/bin/env bash
# demo.sh: [NAME] [extra launch args]: the full demo with the YOLO detector
NAME=${1:-yolo_demo_$(date +%H%M)}; shift
W=${YOLO_WEIGHTS:-$HOME/rs1_18_ws/eval/yolo/weights/evidence_v1.pt}
[ -f "$W" ] || { echo "no YOLO weights at $W (train them: tools/yolo/README.md)"; exit 1; }
[ -d "${YOLO_PKGS:-$HOME/yolo_pkgs}/ultralytics" ] || { echo "no ~/yolo_pkgs (tools/yolo/README.md, step 1)"; exit 1; }
ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-7} exec bash "$(dirname "$0")/../eval/run_removal_test.sh" "$NAME" \
  world:=showcase_forest n_trees:=4 balanced:=true use_husky:=true waste:=true \
  yolo:=true evidence:=true settle_loops:=2 yolo_weights:="$W" "$@"
