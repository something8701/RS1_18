#!/usr/bin/env bash
# results.sh: [--open]: the finished run's evidence, for the testing part of the demo: the removal-test report …
source "$(dirname "$0")/demo_env.sh"
NAME=$(current_run); L=$(current_log)
R=$(grep -oE "/tmp/deforestation_eval/removal_test_[0-9_]+\.md" "$L" 2>/dev/null | tail -1)
if [ -z "$R" ] || [ ! -f "$R" ]; then
  warn "no report yet for $NAME (the run is still going: tools/demo/watch.sh)"
else
  mkdir -p "$DEMO_DIR/$NAME"
  cp "$R" "$DEMO_DIR/$NAME/report.md"; cp "${R%.md}.json" "$DEMO_DIR/$NAME/report.json"
  say "Report of $NAME"
  grep -E "^\*\*|^\| |PASS|FAIL" "$R" | sed 's/\*\*//g' | head -40
  python3 - "$DEMO_DIR/$NAME/report.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
s = d.get('evidence_score')
if s:
    print('\nYOLO evidence (found / placed, false finds, mean position error):')
    for k, v in s.items():
        print(f"  {k:6} {v['found']}/{v['total']}   false {v['false']}   error {v['mean_err_m']} m")
EOF
fi

SHEET=$DEMO_DIR/$NAME/photos.jpg
python3 - "$SHEET" <<'EOF'
import glob, os, sys
import cv2
import numpy as np
paths = (sorted(glob.glob('/tmp/deforestation_eval/evidence_photos/*.jpg'), key=os.path.getmtime)[-8:]
         + sorted(glob.glob('/tmp/deforestation_eval/waste_photos/*.jpg'), key=os.path.getmtime)[-4:])
if not paths:
    sys.exit(0)
tiles = []
for p in paths:
    im = cv2.resize(cv2.imread(p), (480, 320))
    cv2.putText(im, os.path.basename(p)[:-4], (8, 312), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    tiles.append(im)
while len(tiles) % 4:
    tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[1], np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
print(f'\nPhotos: {sys.argv[1]} ({len(paths)} photos)')
EOF

say "Earlier runs (the same system under other conditions): $SRC/DEMO_PLAN.md §4"
if [ "$1" = "--open" ]; then
  [ -f "$SHEET" ] && (xdg-open "$SHEET" > /dev/null 2>&1 &)
  [ -f "$DEMO_DIR/$NAME/report.md" ] && (xdg-open "$DEMO_DIR/$NAME/report.md" > /dev/null 2>&1 &)
fi
