#!/usr/bin/env bash
# stop.sh: End the demo run cleanly (the dashboard recording first, so the video is closed properly; then …
source "$(dirname "$0")/demo_env.sh"
NAME=$(current_run)
if [ -n "$NAME" ] && [ -d "$DEMO_DIR/$NAME" ] && pgrep -f "[r]ecord_dashboard.py" > /dev/null; then
  touch "$DEMO_DIR/$NAME/stop"
  for _ in $(seq 1 30); do pgrep -f "[r]ecord_dashboard.py" > /dev/null || break; sleep 1; done
  ok "recording closed: $DEMO_DIR/$NAME/dashboard.mp4"
fi
bash "$SRC/tools/eval/stop_sim.sh"
