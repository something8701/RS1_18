"""Area-alert rules offline: which baseline-canopy fraction / cluster size
gives canopy-LOST clusters at removed trees but none far from them.

usage: area_rules.py BAG:WORLD:T_REMOVED:TARGETS(comma names|'-') ...
"""
import math
import sys

import numpy as np
from scipy import ndimage
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from deforestation_monitoring.replay_evaluate import scan_to_world
from deforestation_monitoring.tree_detection import parse_tree_truth
from deforestation_monitoring.visibility import classify_tree, removal_tier

RES, ORIGIN, DIM, CANOPY = 0.25, -40.0, 320, 2.0
FRACS = (0.6, 0.5, 0.4, 0.3)
SIZES = (8, 5)
W = '/home/alig/rs1_18_ws/src/RS1_18/41068_ignition_bringup/worlds/'
xs = ORIGIN + (np.arange(DIM) + 0.5) * RES


def run(bag, world, t_rem, names):
    truth = parse_tree_truth(W + world + '.sdf')
    by = {t[0]: t for t in truth}
    removed = [by[n] for n in names]
    touch = np.zeros((DIM, DIM), np.int32)
    creads = np.zeros((DIM, DIM), np.int32)
    low = np.zeros((DIM, DIM), np.int16)
    recent = np.full((DIM, DIM), -1.0, np.float32)
    frac = pre = None
    alerts = {k: {} for k in [(f, n) for f in FRACS for n in SIZES]}
    r = SequentialReader()
    r.open(StorageOptions(uri=bag, storage_id='sqlite3'),
           ConverterOptions('cdr', 'cdr'))
    odom, k = None, 0
    while r.has_next():
        topic, data, ts = r.read_next()
        if topic == '/parrot1/odometry':
            odom = deserialize_message(data, Odometry)
            continue
        if topic != '/parrot1/scan' or odom is None:
            continue
        pts = scan_to_world(deserialize_message(data, LaserScan), odom, 10.0)
        ix = ((pts[:, 0] - ORIGIN) / RES).astype(int)
        iy = ((pts[:, 1] - ORIGIN) / RES).astype(int)
        ok = (ix >= 0) & (ix < DIM) & (iy >= 0) & (iy < DIM)
        if not ok.any():
            continue
        sm = np.full((DIM, DIM), -np.inf, np.float32)
        np.maximum.at(sm, (ix[ok], iy[ok]), pts[ok, 2])
        t = np.isfinite(sm)
        h = np.maximum(sm[t], 0.0)
        recent[t] = h
        if frac is None:
            touch[t] += 1
            creads[t] += h > CANOPY
            if ts / 1e9 >= t_rem:
                frac = creads / np.maximum(touch, 1)
                well = touch >= 3
                pre = recent.copy()
            continue
        low[t] = np.where(h < CANOPY, low[t] + 1, 0)
        k += 1
        if k % 10:
            continue
        for (f, n), store in alerts.items():
            mask = well & (frac >= f) & (low >= 3)
            lab, m = ndimage.label(mask, structure=np.ones((3, 3)))
            if not m:
                continue
            sizes = ndimage.sum(mask, lab, range(1, m + 1))
            cents = ndimage.center_of_mass(mask, lab, range(1, m + 1))
            for sz, (cx, cy) in zip(sizes, cents):
                if sz >= n:
                    wx, wy = ORIGIN + (cx + .5) * RES, ORIGIN + (cy + .5) * RES
                    store[(round(wx), round(wy))] = (wx, wy)
    tiers = {t[0]: removal_tier(t, classify_tree(pre, RES, (ORIGIN, ORIGIN), t)[0], truth)
             for t in removed}
    out = {}
    for key, store in alerts.items():
        pts = list(store.values())
        hit = [t[0] for t in removed
               if any(math.hypot(p[0] - t[1], p[1] - t[2]) <= 3.0 for p in pts)]
        false = [p for p in pts if all(math.hypot(p[0] - t[1], p[1] - t[2]) > 5.5
                                       for t in removed)]
        out[key] = (hit, false)
    return out, tiers


for spec in sys.argv[1:]:
    bag, world, t_rem, names = spec.split(':')
    names = [] if names == '-' else names.split(',')
    out, tiers = run(bag, world, float(t_rem), names)
    print(f'\n{bag}  tiers: {tiers}')
    for (f, n), (hit, false) in out.items():
        cs = [h for h in hit if tiers.get(h) in ('crown-shared', 'understory')]
        print(f'  frac>={f} cluster>={n}: area alerts at {len(hit)}/{len(names)} removed '
              f'(crown-shared/understory: {cs}), FALSE area alerts (> 5.5 m from any removal): {len(false)}')
