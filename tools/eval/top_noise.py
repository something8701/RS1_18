"""|top after - top before| over trunks of STANDING trees (the noise floor)."""
import math, sys
import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from nav_msgs.msg import OccupancyGrid
from deforestation_monitoring.tree_detection import parse_tree_truth
from deforestation_monitoring.visibility import disc_max, classify_tree, removal_tier
from deforestation_monitoring.removal_test import select_removal_targets
W = '/home/alig/rs1_18_ws/src/RS1_18/41068_ignition_bringup/worlds/'
def grid(data):
    m = deserialize_message(data, OccupancyGrid)
    g = np.array(m.data, np.float32).reshape(m.info.height, m.info.width).T
    return np.where(g >= 0, g / 10.0, -1.0), m.info.resolution, (m.info.origin.position.x, m.info.origin.position.y)
allv = {}
for spec in sys.argv[1:]:
    bag, world, t_rem, n = spec.split(':'); t_rem = float(t_rem)
    r = SequentialReader(); r.open(StorageOptions(uri=bag, storage_id='sqlite3'), ConverterOptions('cdr', 'cdr'))
    pre = last = None
    while r.has_next():
        topic, data, ts = r.read_next()
        if topic != '/forest_canopy_map': continue
        if ts / 1e9 <= t_rem: pre = data
        last = data
    g0, g1 = grid(pre), grid(last)
    truth = parse_tree_truth(W + world + '.sdf')
    rem = select_removal_targets(truth, int(n), (-30, 30, -30, 30), balanced=(world == 'showcase_forest'))
    for t in truth:
        if abs(t[1]) > 28 or abs(t[2]) > 28 or any(math.hypot(t[1]-q[1], t[2]-q[2]) < 8 for q in rem): continue
        cls = removal_tier(t, classify_tree(*g0, t)[0], truth)
        a, b = disc_max(*g0, t[1], t[2], 0.75), disc_max(*g1, t[1], t[2], 0.75)
        if a is None or b is None: continue
        allv.setdefault(cls, []).append(b - a)
for cls, v in allv.items():
    v = np.array(v)
    print(f'{cls:13s} n={v.size:4d}  |d| p50 {np.percentile(abs(v),50):.2f} p90 {np.percentile(abs(v),90):.2f} p95 {np.percentile(abs(v),95):.2f} p99 {np.percentile(abs(v),99):.2f} max {abs(v).max():.2f}   drops > 0.5: {(v < -0.5).sum()}  > 0.8: {(v < -0.8).sum()}')
