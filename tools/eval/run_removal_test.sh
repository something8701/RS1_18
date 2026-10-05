#!/usr/bin/env bash
# run_removal_test.sh NAME [extra launch args]: run removal test + bag, print report, stop.
# EXTRA_TOPICS="..." adds topics to the recording (e.g. the drone camera).
S=${EVAL_DIR:-$HOME/rs1_18_ws/eval}; mkdir -p $S/logs
# A previous run's stop_sim.sh may still be in its kill pass; it would take
# this run's fresh processes with it (it matches every sim process).
until ! ps -eo args | grep -q "^bash .*tools/eval/stop_sim.sh"; do sleep 1; done
N=$1; shift; L=$S/logs/$N.log
cd ~/rs1_18_ws && export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0} ROS_LOCALHOST_ONLY=1 ROS_LOG_DIR=$S/logs/ros DISPLAY=:1
source /opt/ros/humble/setup.bash && source install/setup.bash
rm -rf ~/rs1_18_ws/bags/$N
(setsid nohup ros2 launch deforestation_monitoring removal_test.launch.py "$@" > $L 2>&1 < /dev/null &)
sleep 15
(cd ~/rs1_18_ws/bags && setsid nohup ros2 bag record -o $N /parrot1/scan /parrot1/odometry /forest_canopy_map /forest_canopy_hits /canopy_change_map /parrot_tree_change_events /parrot_tree_baseline /parrot_tree_positions /canopy_change_events /scan_coverage /suspicious_areas /survey_status $EXTRA_TOPICS > $S/logs/bag_$N.log 2>&1 < /dev/null &)
until grep -qE "\[test\] (PASS|FAIL)|max_wait_s exceeded|removal_test.*process has died" $L; do sleep 10; done
sleep 3
grep -E "\[report\]" $L | sed 's/.*\[report\] //'
R=$(ls -t /tmp/deforestation_eval/removal_test_*.md | head -1); cp $R $S/logs/$N.report.md; cp ${R%.md}.json $S/logs/$N.report.json
$(dirname "$0")/stop_sim.sh
