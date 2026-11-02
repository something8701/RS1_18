#!/usr/bin/env bash
# preflight.sh: [--clean]: check everything the live demo needs, before the tutors arrive
source "$(dirname "$0")/demo_env.sh"
FAIL=0
fail() { bad "$*"; FAIL=1; }
say "Demo preflight"

if pgrep -f "[c]ollect_dataset.py" > /dev/null; then
  fail "a YOLO dataset collection is running; --clean would stop its Gazebo. Wait for it (or tools/yolo/stop_gz.sh)"
  echo; say "NOT READY"; exit 1
fi
pgrep -f "[t]ools/yolo/train.py" > /dev/null && warn "YOLO training is running (the GPU is busy; the sim will be slower)"

SIMPAT='[i]gn gazebo|[r]os2 launch deforestation_monitoring|[r]osbridge_websocket'
if pgrep -f "$SIMPAT" > /dev/null; then
  if [ "$1" = "--clean" ]; then
    bash "$SRC/tools/eval/stop_sim.sh" > /dev/null; bash "$SRC/tools/yolo/stop_gz.sh"
    pgrep -f "$SIMPAT" > /dev/null && fail "processes still running after --clean" || ok "stopped a leftover simulation"
  else
    fail "a simulation is still running (run: tools/demo/preflight.sh --clean)"
  fi
else ok "no simulation running"; fi
for port in 8081 9090; do
  if ss -ltn 2>/dev/null | grep -q ":$port "; then fail "port $port is in use"; else ok "port $port free"; fi
done

ros_env 2>/dev/null
[ -f "$WS/install/setup.bash" ] && ok "workspace built" || fail "no install/ (colcon build)"
ros2 pkg prefix deforestation_monitoring > /dev/null 2>&1 && ok "deforestation_monitoring found" \
  || fail "deforestation_monitoring not built"
python3 -c "from deforestation_interfaces.msg import EvidenceArray" 2>/dev/null \
  && ok "interfaces up to date (EvidenceArray)" || fail "rebuild deforestation_interfaces"
NEWER=$(find "$SRC/deforestation_monitoring/setup.py" "$SRC/deforestation_interfaces/msg" \
  -newer "$WS/build/deforestation_monitoring/colcon_build.rc" 2>/dev/null | head -1)
[ -z "$NEWER" ] && ok "build newer than setup.py and messages" \
  || warn "changed since the last build: $NEWER (colcon build --symlink-install)"

YP=${YOLO_PKGS:-$HOME/yolo_pkgs}
[ -d "$YP/ultralytics" ] && ok "YOLO packages in $YP" || fail "no $YP (tools/yolo/README.md step 1)"
[ -f "$YOLO_WEIGHTS" ] && ok "weights $(basename "$YOLO_WEIGHTS")" || fail "no weights $YOLO_WEIGHTS"
PYTHONPATH=$YP:$PYTHONPATH python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null \
  && ok "CUDA available to torch" || warn "no CUDA: YOLO runs on the CPU (slower); set device:=cpu"
ls ~/.local/lib/python3.10/site-packages 2>/dev/null | grep -qi "^onnxruntime$\|^torch$" \
  && warn "torch/onnxruntime in ~/.local (the ROS Python sees it)" || ok "~/.local clean"

nvidia-smi -L > /dev/null 2>&1 && ok "GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)" \
  || warn "nvidia-smi failed"
FREE=$(df -BG --output=avail "$WS" | tail -1 | tr -dc 0-9)
[ "$FREE" -gt 20 ] && ok "disk: ${FREE} GB free" || fail "disk: only ${FREE} GB free"
xdpyinfo -display "$DISPLAY" > /dev/null 2>&1 && ok "display $DISPLAY" || warn "display $DISPLAY not reachable (Gazebo window)"
command -v xdg-open > /dev/null && ok "xdg-open (opens the dashboard)" || warn "no xdg-open: open $DASH by hand"

if [ "$1" = "--clean" ]; then
  rm -rf /tmp/deforestation_eval/evidence_photos /tmp/deforestation_eval/waste_photos /tmp/deforestation_inspections
  ok "cleared the photo folders of earlier runs"
fi

BK=$(ls -t "$DEMO_DIR"/*/dashboard.mp4 "$WS"/recordings/*/dashboard.mp4 2>/dev/null | head -1)
[ -n "$BK" ] && ok "backup video: $BK" || warn "no backup video (tools/demo/record_backup.sh)"

echo
[ $FAIL = 0 ] && say "READY. Start the demo: tools/demo/start.sh" || say "NOT READY: fix the ✘ lines above."
exit $FAIL
