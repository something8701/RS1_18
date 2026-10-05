#!/usr/bin/env bash
# run_showcase_recording.sh OUT_DIR [extra launch args]: one removal test,
# fully recorded for showing: rosbag (incl. the drone camera and the
# dashboard's camera topics), a dashboard-only video (time-lapse, ends on the
# Changes view and the report), the launch log and the removal-test report.
# Then run demo_replay on OUT_DIR/bag for replay.mp4 + comparison.png.
#   WORLD=dense_forest (default) | showcase_forest ...
#   SPEED=10 (time-lapse factor of the dashboard video, captured at 1 fps)
OUT=$(realpath -m "$1"); shift
WORLD=${WORLD:-dense_forest}; SPEED=${SPEED:-10}
mkdir -p "$OUT"; L=$OUT/launch.log
# A previous run's stop_sim.sh may still be in its kill pass; it would take
# this run's fresh processes with it (it matches every sim process).
until ! ps -eo args | grep -q "^bash .*tools/eval/stop_sim.sh"; do sleep 1; done
cd ~/rs1_18_ws && export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-7} ROS_LOCALHOST_ONLY=1 ROS_LOG_DIR=$OUT/ros DISPLAY=:1
source /opt/ros/humble/setup.bash && source install/setup.bash
rm -rf "$OUT/bag" "$OUT/stop" "$OUT/finale"
(setsid nohup ros2 launch deforestation_monitoring removal_test.launch.py world:=$WORLD "$@" > "$L" 2>&1 < /dev/null &)
sleep 15
(cd "$OUT" && setsid nohup ros2 bag record -o bag /parrot1/scan /parrot1/odometry /forest_canopy_map \
  /forest_canopy_hits /canopy_change_map /parrot_tree_change_events /parrot_tree_baseline \
  /parrot_tree_positions /canopy_change_events /scan_coverage /suspicious_areas /survey_status \
  /parrot_tree_baseline_info /parrot_tree_change_notes /parrot1/camera/image \
  /parrot1/camera/depth/image /clock > "$OUT/bag.log" 2>&1 < /dev/null &)
until curl -s -o /dev/null http://localhost:8081/api/data; do sleep 2; done
python3 src/RS1_18/tools/eval/record_dashboard.py --out "$OUT/dashboard.mp4" --fps 1 \
  --speed "$SPEED" --duration 3600 --finale-file "$OUT/finale" --finale-s 6 > "$OUT/recorder.log" 2>&1 &
REC=$!
until grep -qE "\[test\] (PASS|FAIL)|max_wait_s exceeded|removal_test.*process has died" "$L"; do sleep 10; done
sleep 20                                   # the final map, live, before the finale
touch "$OUT/finale"; wait $REC
grep -E "\[report\]" "$L" | sed 's/.*\[report\] //' | tee "$OUT/summary.txt"
R=$(ls -t /tmp/deforestation_eval/removal_test_*.md | head -1)
cp "$R" "$OUT/report.md"; cp "${R%.md}.json" "$OUT/report.json"
$(dirname "$0")/stop_sim.sh
