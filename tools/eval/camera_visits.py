"""Simulate the tracker's camera see-through rule per frame on a bag (option A).

usage: camera_visits.py BAG WORLD [--log LAUNCH_LOG] [--gap 30] [--min-rays 4]
                        [--weighted] [--open 0.3] [--radius 0.75]

For every tracker baseline position in the bag (/parrot_tree_baseline) and
every removed SDF trunk, each camera frame gives a canopy fraction over the
disc: canopy pixels >= 3 m / (those + rays passing 3 m open), from the same
camera_species code as the live node. Frames with fewer than --min-rays rays
in the disc are skipped. Frames closer than --gap s form one visit; a visit's
mean is ray-weighted (--weighted) or the mean of its frame fractions.

Printed: for removed trees, the visits after the cut and when the second
low visit in a row ends (the camera evidence time); for standing baseline
positions, how many ever have two low visits in a row (candidates the
height-drop and unmatched conditions must still stop).
"""
import argparse
import gzip
import math
import re

import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, PointCloud2

from deforestation_monitoring.camera_species import cell_counts, open_rays, project, quat_to_matrix
from deforestation_monitoring.removal_test import _cloud_xyz
from deforestation_monitoring.tree_detection import parse_tree_truth


RES, ORIGIN, DIM = 0.25, -40.0, 320


def disc_cells(x, y, radius):
    """Flat [ix, iy] indices of the grid cells whose centres are within radius."""
    xs = ORIGIN + (np.arange(DIM) + 0.5) * RES
    ix = np.nonzero(np.abs(xs - x) <= radius + RES)[0]
    iy = np.nonzero(np.abs(xs - y) <= radius + RES)[0]
    gx, gy = np.meshgrid(ix, iy, indexing='ij')
    inside = np.hypot(xs[gx] - x, xs[gy] - y) <= radius
    return (gx[inside] * DIM + gy[inside]).ravel()


def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def read(bag):
    r = SequentialReader()
    r.open(StorageOptions(uri=bag, storage_id='sqlite3'), ConverterOptions('cdr', 'cdr'))
    odoms, rgbs, depths, base, recv = [], {}, {}, None, []
    while r.has_next():
        topic, data, ts = r.read_next()
        if topic == '/parrot1/odometry':
            m = deserialize_message(data, Odometry)
            odoms.append((stamp(m), m))
            recv.append((ts * 1e-9, stamp(m)))
        elif topic == '/parrot1/camera/image':
            m = deserialize_message(data, Image)
            rgbs[round(stamp(m), 3)] = m
        elif topic == '/parrot1/camera/depth/image':
            m = deserialize_message(data, Image)
            depths[round(stamp(m), 3)] = m
        elif topic == '/parrot_tree_baseline' and base is None:
            base = [(float(x), float(y)) for x, y, _ in
                    _cloud_xyz(deserialize_message(data, PointCloud2))]
    return odoms, rgbs, depths, base or [], recv


def removal(log, recv):
    opener = gzip.open if log.endswith('.gz') else open
    m = re.search(r'\[(\d+\.\d+)\] \[removal_test\]: \[test\] removed \d+: (.*)',
                  opener(log, 'rt', errors='replace').read())
    wall = float(m.group(1))
    w, s = min(recv, key=lambda p: abs(p[0] - wall))
    return s + (wall - w), [n.strip() for n in m.group(2).split(',')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag')
    ap.add_argument('world')
    ap.add_argument('--log')
    ap.add_argument('--gap', type=float, default=30.0)
    ap.add_argument('--min-rays', type=int, default=4)
    ap.add_argument('--weighted', action='store_true')
    ap.add_argument('--open', type=float, default=0.3)
    ap.add_argument('--radius', type=float, default=0.75)
    ap.add_argument('--height', type=float, default=3.0)
    a = ap.parse_args()
    odoms, rgbs, depths, base, recv = read(a.bag)
    truth = {n: (x, y) for n, x, y in parse_tree_truth(a.world)}
    t_cut, removed = removal(a.log, recv) if a.log else (None, [])
    sites = {f'base#{i + 1}': p for i, p in enumerate(base)}
    sites.update({n: truth[n] for n in removed})
    pos = np.array(list(sites.values()))
    names = list(sites)
    odom_t = np.array([t for t, _ in odoms])
    frames = {n: [] for n in names}               # (t, high, open) per site
    discs = [disc_cells(px, py, a.radius) for px, py in pos]
    grid = (RES, ORIGIN, ORIGIN, DIM, DIM)
    for t in sorted(set(rgbs) & set(depths)):
        i = int(np.argmin(np.abs(odom_t - t)))
        if abs(odom_t[i] - t) > 0.1:
            continue
        m, d, o = rgbs[t], depths[t], odoms[i][1]
        rgb = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
        depth = np.frombuffer(d.data, np.float32).reshape(d.height, d.width)
        q = o.pose.pose.orientation
        rot = quat_to_matrix(q.x, q.y, q.z, q.w)
        x, y = o.pose.pose.position.x, o.pose.pose.position.y
        pts, _ = project(rgb, depth, rot, x, y, 10.0)
        hc = np.zeros(DIM * DIM)
        oc = np.zeros(DIM * DIM)
        cells, n = cell_counts(pts[pts[:, 2] >= a.height, :2], *grid)
        hc[cells] = n
        cells, n = cell_counts(open_rays(depth, rot, x, y, 10.0, a.height), *grid)
        oc[cells] = n
        near = np.hypot(pos[:, 0] - x, pos[:, 1] - y) < 25.0
        for k in np.nonzero(near)[0]:
            h, op = hc[discs[k]].sum(), oc[discs[k]].sum()
            if h + op >= a.min_rays:
                frames[names[k]].append((t, h, op))

    def visits(fr):
        out, cur = [], []
        for f in fr:
            if cur and f[0] - cur[-1][0] > a.gap:
                out.append(cur)
                cur = []
            cur.append(f)
        if cur:
            out.append(cur)
        res = []
        for v in out:
            hs, os_ = np.array([f[1] for f in v]), np.array([f[2] for f in v])
            mean = (hs.sum() / (hs + os_).sum() if a.weighted
                    else float(np.mean(hs / (hs + os_))))
            res.append((v[0][0], v[-1][0], mean))
        return res

    print(f'rule: gap {a.gap} s, min rays {a.min_rays}, '
          f'{"ray-weighted" if a.weighted else "frame-mean"}, open < {a.open}')
    if removed:
        print('removed trees (visits after the cut: start+s mean; evidence = end of 2nd low visit in a row):')
        for n in removed:
            vs = [v for v in visits(frames[n]) if v[0] > t_cut]
            ev = next((vs[i][1] - t_cut for i in range(1, len(vs))
                       if vs[i][2] < a.open and vs[i - 1][2] < a.open), None)
            seen = ' '.join(f'{v[0] - t_cut:+.0f}s:{v[2]:.2f}' for v in vs)
            verdict = f'evidence at {ev:.0f} s' if ev is not None else 'no evidence'
            print(f'  {n:10s} {seen}  -> {verdict}')
    flagged = []
    for n in names:
        if not n.startswith('base#'):
            continue
        p = sites[n]
        if removed and min(math.hypot(p[0] - truth[r][0], p[1] - truth[r][1])
                           for r in removed) < 6.0:
            continue                               # near a cut: a drop is real
        vs = visits(frames[n])
        pairs = [i for i in range(1, len(vs)) if vs[i][2] < a.open and vs[i - 1][2] < a.open]
        if pairs:
            flagged.append((n, p, [round(v[2], 2) for v in vs]))
    far = sum(1 for n in names if n.startswith('base#') and not (removed and min(
        math.hypot(sites[n][0] - truth[r][0], sites[n][1] - truth[r][1]) for r in removed) < 6.0))
    print(f'standing baseline positions > 6 m from any cut: {far}; '
          f'with two low visits in a row: {len(flagged)}')
    for n, p, v in flagged[:8]:
        print(f'  {n} ({p[0]:.1f}, {p[1]:.1f}) visits {v}')


if __name__ == '__main__':
    main()
