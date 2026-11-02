#!/usr/bin/env bash
# backup.sh: [VIDEO]: play the backup recording (if the live run fails)
source "$(dirname "$0")/demo_env.sh"
V=${1:-$(ls -t "$DEMO_DIR"/*/dashboard.mp4 "$WS"/recordings/*/dashboard.mp4 2>/dev/null | head -1)}
[ -f "$V" ] || { bad "no backup video found"; exit 1; }
ok "playing $V"
(xdg-open "$V" > /dev/null 2>&1 &)
R=$(dirname "$V")/report.md
[ -f "$R" ] && { say "Its report:"; grep -E "^# |Right:|Husky|YOLO" "$R" | sed 's/\*\*//g'; }
