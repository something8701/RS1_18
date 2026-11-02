#!/usr/bin/env bash
# watch.sh: [RUN_NAME]: follow the demo run live, one line per milestone with the run time, and a cue (») for …
source "$(dirname "$0")/demo_env.sh"
NAME=${1:-$(current_run)}
L=$DEMO_DIR/$NAME.log
[ -f "$L" ] || { echo "no log $L (start.sh first)"; exit 1; }
say "Watching $NAME  ($L)"
tail -n +1 -F "$L" 2>/dev/null | awk '
function t(line,   m) {
  if (match(line, /\[1[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]\./)) { s = substr(line, RSTART+1, 10) + 0; if (!t0) t0 = s;
    return sprintf("%02d:%02d", int((s-t0)/60), (s-t0)%60) }
  return "  -  " }
function msg(line) { sub(/^.*\]: /, "", line); return line }
function out(col, line, text) { printf "\033[%sm%s  %s\033[0m\n", col, t(line), text; fflush() }
function cue(text) { printf "\033[1;36m       » %s\033[0m\n", text; fflush() }
/Removal test ready/ { out("1", $0, "Simulation up. " msg($0));
  cue("Say what the system does in one sentence, then the architecture while the drone surveys.") }
/\[BASELINE\] Scan/ && match($0, /\(([0-9.]+)% of survey area\)/) {
  p = substr($0, RSTART+1, RLENGTH-17) + 0; b = int(p/20)*20;
  if (b > last_cov) { last_cov = b; out("0", $0, "Drone survey: " p "% covered") } }
/Parrot tree baseline frozen/ { out("1;32", $0, msg($0));
  cue("Baseline frozen: the drone now knows every tree. Next the test cuts 4 trees.") }
/\[test\] removed/ { out("1;33", $0, "TREES CUT: " msg($0));
  cue("A stump, a felled log and a red bag are left at each. Show the change map; LOST events come next.") }
/Parrot tree event: LOST/ { out("31", $0, msg($0)) }
/queued @/ { if (!q++) cue("Areas are queued for the Husky. Interactive: tools/demo/husky.sh ask on (tutor picks the area)");
  out("33", $0, msg($0)) }
/Searching / { m = msg($0); sub(/: waypoint.*/, "", m); if (m != last_search) { last_search = m; out("36", $0, "Husky: " m) } }
/Searching / && !shown_cam++ { cue("The Husky is searching: show its camera (YOLO boxes) and the 3D view.") }
/SITE INSPECTED/ { m = msg($0); sub(/ \[image:.*/, "", m); out("1;36", $0, m) }
/WASTE #.*found by the Husky/ { out("1;35", $0, msg($0)) }
/EVIDENCE / { m = msg($0); sub(/, photo .*/, "", m); out("35", $0, "YOLO " m) }
/Operator: / { out("1;34", $0, msg($0)) }
/Nav2 server not ready/ && !navwarn++ { out("1;31", $0, "THE HUSKY CANNOT DRIVE: its navigation (Nav2) did not start.");
  cue("Restart the run: tools/demo/stop.sh && tools/demo/start.sh (or play the backup: demo backup)") }
/process has died/ && !finished { out("1;31", $0, "CRASH: " msg($0)) }
/\[test\] (PASS|FAIL)/ { finished = 1; out("1;32", $0, msg($0));
  cue("Run finished: tools/demo/results.sh (report, scores, photos). The sim stays up for questions.") }
'
