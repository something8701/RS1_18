#!/usr/bin/env bash
# run_gate1.sh NAME [HOLD_S]: no-change gate. Launch the demo with nothing
# removed, record, wait for the tree baseline to freeze, hold HOLD_S seconds
# (default 360 = ~3 sparse survey loops), then report baseline vs SDF and
# every change event.
#   WORLD=sparse_trees (default): GATE 1, PASS = 5/0/0 and zero events.
#   WORLD=cluster_test / dense_forest: GATE 2 / GATE 3, PASS = zero events
#   (baseline quality is reported, not gated).
#   EXTRA_TOPICS="..." adds topics to the recording (e.g. the camera).
#   LAUNCH_ARGS="..." adds launch arguments (e.g. colour_lost:=true).
S=${EVAL_DIR:-$HOME/rs1_18_ws/eval}; mkdir -p $S/logs
# A previous run's stop_sim.sh may still be in its kill pass; it would take
# this run's fresh processes with it (it matches every sim process).
until ! ps -eo args | grep -q "^bash .*tools/eval/stop_sim.sh"; do sleep 1; done
N=$1; HOLD=${2:-360}; L=$S/logs/$N.log; WORLD=${WORLD:-sparse_trees}
cd ~/rs1_18_ws && export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0} ROS_LOCALHOST_ONLY=1 ROS_LOG_DIR=$S/logs/ros DISPLAY=:1
source /opt/ros/humble/setup.bash && source install/setup.bash
rm -rf ~/rs1_18_ws/bags/$N
(setsid nohup ros2 launch deforestation_monitoring demo_full.launch.py world:=$WORLD remove_trees:=false $LAUNCH_ARGS > $L 2>&1 < /dev/null &)
sleep 15
(cd ~/rs1_18_ws/bags && setsid nohup ros2 bag record -o $N /parrot1/scan /parrot1/odometry /forest_canopy_map /canopy_change_map /parrot_tree_baseline /parrot_tree_positions /parrot_tree_change_events /canopy_change_events /scan_coverage /suspicious_areas $EXTRA_TOPICS > $S/logs/bag_$N.log 2>&1 < /dev/null &)
until grep -qE "baseline frozen|process has died" $L; do sleep 5; done
sleep $HOLD
$(dirname "$0")/stop_sim.sh
python3 - "$N" "$L" "$WORLD" <<'PY'
import math, re, sys
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from sensor_msgs.msg import PointCloud2
from deforestation_monitoring.removal_test import _cloud_xyz, baseline_quality
from deforestation_monitoring.tree_detection import parse_tree_truth
name, log, world = sys.argv[1:4]
half = {'sparse_trees': 15, 'cluster_test': 10}.get(world, 30)
truth = parse_tree_truth(f'/home/alig/rs1_18_ws/src/RS1_18/41068_ignition_bringup/worlds/{world}.sdf')
r = SequentialReader()
r.open(StorageOptions(uri=f'/home/alig/rs1_18_ws/bags/{name}', storage_id='sqlite3'), ConverterOptions('cdr', 'cdr'))
base, t_f, t_end = None, None, None
while r.has_next():
    topic, data, ts = r.read_next(); t_end = ts / 1e9
    if topic == '/parrot_tree_baseline' and base is None:
        base = [(int(round(z)), float(x), float(y)) for x, y, z in _cloud_xyz(deserialize_message(data, PointCloud2))]
        t_f = ts / 1e9
txt = open(log, errors='replace').read()
tree_ev = re.findall(r'Parrot tree event: (LOST|GAINED) .*', txt)
canopy_ev = re.findall(r'(CANOPY LOST|NEW CANOPY): .*', txt)
q = baseline_quality(truth, base or [], (-half, half, -half, half))
no_events = not tree_ev and not canopy_ev
if world == 'sparse_trees':
    gate, ok = 'GATE 1', (q['tp'], q['fp'], q['fn']) == (5, 0, 0) and no_events
else:
    gate, ok = {'cluster_test': 'GATE 2'}.get(world, 'GATE 3'), no_events
print(f"{gate} ({world}) {'PASS' if ok else 'FAIL'}: baseline TP/FP/FN {q['tp']}/{q['fp']}/{q['fn']}; "
      f"observed {t_end - t_f:.0f} s after freeze; "
      f"tree events {len(tree_ev)}, canopy events {len(canopy_ev)}")
for e in re.findall(r'Parrot tree event: .*|CANOPY LOST: .*|NEW CANOPY: .*', txt):
    print('  ', e[:160])
PY
