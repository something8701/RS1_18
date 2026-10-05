"""Would a max-height CHM give better baseline positions than the live map?

usage: maxchm_positions.py BAG WORLD

The live /forest_canopy_map keeps the latest reading per cell, so every
drone pass reshapes crown edges and detections drift (dense replays: oak_153
within 1 m early in the survey, 2.1 m at the freeze). This rebuilds the CHM
from the bag's scans up to the tracker's freeze (the first
/parrot_tree_baseline message) with the all-time maximum per cell
(replay_evaluate.accumulate), runs the live detector on it and on the
latest-reading CHM at the same moment (same resolution and parameters), and
compares, per canopy-tier tree in the survey box, the distance to the
nearest detection on each with the live baseline's.
"""
import math
import sys

import numpy as np
import yaml
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from sensor_msgs.msg import PointCloud2

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from deforestation_monitoring.replay_evaluate import accumulate, scan_to_world
from deforestation_monitoring.removal_test import _cloud_xyz
from deforestation_monitoring.tree_detection import DetectionParams, detect_trees, parse_tree_truth
from deforestation_monitoring.visibility import classify_tree, removal_tier

OAK_H, SNAP = 5.5, 2.0
PARAMS = '/home/alig/rs1_18_ws/src/RS1_18/deforestation_monitoring/config/tree_detection_params.yaml'


def reader(bag):
    r = SequentialReader()
    r.open(StorageOptions(uri=bag, storage_id='sqlite3'), ConverterOptions('cdr', 'cdr'))
    return r


def mean_chm(r, dim=320, res=0.25, origin=-40.0):
    """Time average of the latest-reading CHM: every 1 s of bag time, add the
    current latest reading of every scanned cell (what the tracker would see
    each tick). Returns (mean, count)."""
    recent = np.zeros((dim, dim), np.float32)
    seen = np.zeros((dim, dim), bool)
    total = np.zeros((dim, dim), np.float64)
    n = np.zeros((dim, dim), np.float64)
    odom, next_tick = None, None
    while r.has_next():
        topic, data, ts = r.read_next()
        if topic == '/parrot1/odometry':
            odom = deserialize_message(data, Odometry)
        elif topic == '/parrot1/scan' and odom is not None:
            pts = scan_to_world(deserialize_message(data, LaserScan), odom, 10.0)
            if len(pts):
                ix = ((pts[:, 0] - origin) / res).astype(int)
                iy = ((pts[:, 1] - origin) / res).astype(int)
                ok = (ix >= 0) & (ix < dim) & (iy >= 0) & (iy < dim)
                scan_max = np.full((dim, dim), -np.inf, np.float32)
                np.maximum.at(scan_max, (ix[ok], iy[ok]), pts[ok, 2])
                hit = np.isfinite(scan_max)
                recent[hit] = np.maximum(scan_max[hit], 0.0)
                seen |= hit
        if next_tick is None:
            next_tick = ts + 10**9
        while ts >= next_tick:
            next_tick += 10**9
            total[seen] += recent[seen]
            n[seen] += 1
    return np.where(n > 0, total / np.maximum(n, 1), 0.0).astype(np.float32), n


class Until:
    """A bag reader that stops at bag time `t_ns`."""

    def __init__(self, r, t_ns):
        self.r, self.t_ns, self.next = r, t_ns, None

    def has_next(self):
        if self.next is None and self.r.has_next():
            self.next = self.r.read_next()
        return self.next is not None and self.next[2] < self.t_ns

    def read_next(self):
        item, self.next = self.next, None
        return item


def main():
    bag, world = sys.argv[1], sys.argv[2]
    r = reader(bag)
    t_freeze, base = None, []
    while r.has_next():
        topic, data, ts = r.read_next()
        if topic == '/parrot_tree_baseline':
            t_freeze = ts
            base = [(float(x), float(y)) for x, y, _ in
                    _cloud_xyz(deserialize_message(data, PointCloud2))]
            break
    max_chm, hits, _, _, _, recent = accumulate(Until(reader(bag), t_freeze), 10.0, res=0.25)
    params = DetectionParams.from_dict(yaml.safe_load(open(PARAMS))['/**']['ros__parameters'])
    params.chm_resolution = 0.25
    scanned = hits > 0
    avg_chm, _ = mean_chm(Until(reader(bag), t_freeze))
    avg_dets = [(d.x, d.y) for d in detect_trees(avg_chm, scanned, params, -40.0, -40.0)]
    max_dets = detect_trees(max_chm, scanned, params, -40.0, -40.0)
    dets = [(d.x, d.y) for d in max_dets]
    recent_dets = detect_trees(recent, scanned, params, -40.0, -40.0)
    dets_recent = [(d.x, d.y) for d in recent_dets]
    # Hybrid: live baseline, but each oak-height tree (latest-CHM max within
    # 0.75 m of its position >= OAK_H) moves to the nearest oak-height max-CHM
    # detection within SNAP m, one-to-one, closest pairs first.
    tall = [d for d in max_dets if d.height >= OAK_H]
    xs = -40.0 + (np.arange(recent.shape[0]) + 0.5) * 0.25
    gx, gy = np.meshgrid(xs, xs, indexing='ij')

    def height_at(x, y):
        disc = (np.hypot(gx - x, gy - y) <= 0.75) & scanned
        return float(recent[disc].max()) if disc.any() else 0.0
    pairs = sorted((math.hypot(t.x - x, t.y - y), i, j) for i, (x, y) in enumerate(base)
                   if height_at(x, y) >= OAK_H for j, t in enumerate(tall)
                   if math.hypot(t.x - x, t.y - y) <= SNAP)
    hybrid, used_b, used_t = list(base), set(), set()
    for _, i, j in pairs:
        if i not in used_b and j not in used_t:
            used_b.add(i)
            used_t.add(j)
            hybrid[i] = (tall[j].x, tall[j].y)
    print(f'hybrid: {len(used_b)} baseline trees moved to a max-CHM oak position')
    truth = [t for t in parse_tree_truth(world) if abs(t[1]) <= 30 and abs(t[2]) <= 30]
    grid = np.where(scanned, max_chm, -1.0)
    rows = []
    for t in truth:
        if removal_tier(t, classify_tree(grid, 0.25, (-40.0, -40.0), t)[0], truth) != 'canopy':
            continue
        dist = [min(math.hypot(x - t[1], y - t[2]) for x, y in ds)
                for ds in (dets, dets_recent, base, hybrid, avg_dets)]
        rows.append((t[0], *dist))
    for kind in ('oak', 'pine'):
        sel = [r for r in rows if r[0].startswith(kind)]
        for lim in (1.0, 2.0):
            print(f'{kind:4s} canopy ({len(sel)}): within {lim} m  max-CHM '
                  f'{sum(r[1] <= lim for r in sel):3d}   latest-CHM {sum(r[2] <= lim for r in sel):3d}'
                  f'   live baseline {sum(r[3] <= lim for r in sel):3d}'
                  f'   hybrid {sum(r[4] <= lim for r in sel):3d}'
                  f'   mean-CHM {sum(r[5] <= lim for r in sel):3d}')
    print(f'detections: max-CHM {len(dets)}, latest-CHM {len(dets_recent)}, live baseline {len(base)}')
    for name in ('oak_153', 'oak_118', 'pine_111', 'pine_163', 'pine_75', 'oak_60', 'oak_68'):
        r = next((r for r in rows if r[0] == name), None)
        if r:
            print(f'  {name:9s} max-CHM {r[1]:.2f}   latest-CHM {r[2]:.2f}   live baseline {r[3]:.2f}'
                  f'   hybrid {r[4]:.2f}   mean-CHM {r[5]:.2f} m')


if __name__ == '__main__':
    main()
