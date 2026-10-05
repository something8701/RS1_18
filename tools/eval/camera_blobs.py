"""Simulate a camera area alert for pines missing from the baseline (option A).

usage: camera_blobs.py BAG WORLD [--log LAUNCH_LOG] [--min-cells 6]
       [--attach] [--site-radius 0.5] [--drop-mode mean --drop 1.0]
       [--show-candidates] [--debug X Y]

The tracker's defaults (area_attach, area_radius 0.5, area_mean_drop 1.0):
    --attach --site-radius 0.5 --drop-mode mean --drop 1.0

At the tracker's freeze (first /parrot_tree_baseline), cells the camera saw
as pine-coloured canopy (mean blue/green >= --pine, canopy fraction >=
--canopy) form patches; a patch of >= --min-cells cells with no baseline
tree within --exclude m is a pine the tracker does not know about (e.g. one
inside an oak's crown). After the freeze each patch gets the tracker's
camera rule: the canopy fraction over a --radius disc at its centre,
averaged per visit (frames split by gaps > --gap s), must be < --open in two
visits in a row, and the LiDAR top there (/forest_canopy_map, max within the
disc) must have dropped >= --drop since the freeze.

Printed: every patch that would raise an alert, with the nearest removed
tree (<= 3 m counts in removal_test) and whether it would be false (> 6 m from
every removed trunk). --drop-mode mean compares the disc mean instead of the
max; --show-candidates prints every site with two see-through visits in a
row, whatever its drop; --debug X Y prints the patches near (X, Y).

--attach: a patch whose nearby trees are all non-pine is watched for the
nearest of them (the tracker's area_attach); when it fires that tree goes
LOST, scored by the identity rule (removed trunk = credited, standing one =
false). --host-max-top limits hosts to trees whose top at the freeze is
below it.
"""
import argparse
import gzip
import math
import re

import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from nav_msgs.msg import OccupancyGrid, Odometry
from scipy import ndimage
from sensor_msgs.msg import Image, PointCloud2

from deforestation_monitoring.camera_species import (
    cell_colour, cell_counts, open_rays, project, quat_to_matrix)
from deforestation_monitoring.removal_test import _cloud_xyz
from deforestation_monitoring.tree_detection import parse_tree_truth

RES, ORIGIN, DIM = 0.25, -40.0, 320
XS = ORIGIN + (np.arange(DIM) + 0.5) * RES
GRID = (RES, ORIGIN, ORIGIN, DIM, DIM)


def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def disc(x, y, radius):
    ix = np.nonzero(np.abs(XS - x) <= radius + RES)[0]
    iy = np.nonzero(np.abs(XS - y) <= radius + RES)[0]
    gx, gy = np.meshgrid(ix, iy, indexing='ij')
    inside = np.hypot(XS[gx] - x, XS[gy] - y) <= radius
    return gx[inside], gy[inside]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag')
    ap.add_argument('world')
    ap.add_argument('--log')
    ap.add_argument('--pine', type=float, default=0.65)
    ap.add_argument('--canopy', type=float, default=0.6)
    ap.add_argument('--min-cells', type=int, default=6)
    ap.add_argument('--exclude', type=float, default=1.5)
    ap.add_argument('--radius', type=float, default=0.75)
    ap.add_argument('--gap', type=float, default=15.0)
    ap.add_argument('--open', type=float, default=0.3)
    ap.add_argument('--drop', type=float, default=0.3)
    ap.add_argument('--site-radius', type=float,
                    help='disc for the patch rule (default --radius; tree colour keeps --radius)')
    ap.add_argument('--attach', action='store_true')
    ap.add_argument('--drop-mode', choices=('max', 'mean'), default='max')
    ap.add_argument('--admit-ticks', type=int, default=0,
                    help='list patches with a /parrot_tree_positions detection within '
                         '--admit-dist m in the last N ticks before the freeze')
    ap.add_argument('--admit-dist', type=float, default=1.0)
    ap.add_argument('--list-patches', action='store_true')
    ap.add_argument('--show-candidates', action='store_true',
                    help='print every site with two see-through visits, whatever its drop')
    ap.add_argument('--host-max-top', type=float, default=99.0)
    ap.add_argument('--debug', type=float, nargs=2)
    a = ap.parse_args()

    r = SequentialReader()
    r.open(StorageOptions(uri=a.bag, storage_id='sqlite3'), ConverterOptions('cdr', 'cdr'))
    odoms, rgbs, depths, maps, base, t_freeze, recv = [], {}, {}, [], None, None, []
    positions = []                                   # pre-freeze detections per tick
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
        elif topic == '/forest_canopy_map':
            m = deserialize_message(data, OccupancyGrid)
            g = np.array(m.data, np.int16).reshape(m.info.height, m.info.width).T
            maps.append((ts * 1e-9, np.where(g >= 0, g / 10.0, -1.0).astype(np.float32)))
        elif topic == '/parrot_tree_positions' and base is None:
            positions.append([(float(x), float(y)) for x, y, _ in
                              _cloud_xyz(deserialize_message(data, PointCloud2))])
        elif topic == '/parrot_tree_baseline' and base is None:
            base = [(float(x), float(y)) for x, y, _ in
                    _cloud_xyz(deserialize_message(data, PointCloud2))]
            t_freeze = ts * 1e-9
    wall_to_sim = np.median([s - w for w, s in recv])
    t_freeze_sim = t_freeze + wall_to_sim
    map_t = np.array([t for t, _ in maps])

    def chm_at(t_sim):
        return maps[max(0, int(np.searchsorted(map_t, t_sim - wall_to_sim)) - 1)][1]

    odom_t = np.array([t for t, _ in odoms])
    shape = (DIM, DIM)
    colour_n, colour_s = np.zeros(shape), np.zeros(shape)
    high, opened = np.zeros(shape), np.zeros(shape)
    after = []                                      # (t, high grid, open grid) after the freeze
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
        pts, cols = project(rgb, depth, rot, x, y, 10.0)
        h = np.zeros(DIM * DIM)
        op = np.zeros(DIM * DIM)
        c, n = cell_counts(pts[pts[:, 2] >= 3.0, :2], *GRID)
        h[c] = n
        c, n = cell_counts(open_rays(depth, rot, x, y, 10.0, 3.0), *GRID)
        op[c] = n
        if t < t_freeze_sim:
            c, n, s = cell_colour(pts, cols, *GRID)
            colour_n.flat[c] += n
            colour_s.flat[c] += s
            high += h.reshape(shape)
            opened += op.reshape(shape)
        else:
            after.append((t, h.reshape(shape), op.reshape(shape)))

    colour = np.where(colour_n > 0, colour_s / np.maximum(colour_n, 1), 0)
    canopy = np.where(high + opened > 0, high / np.maximum(high + opened, 1), 0)
    chm0 = chm_at(t_freeze_sim)

    def disc_mean(grid, has, x, y):
        gx, gy = disc(x, y, a.radius)
        v = grid[gx, gy][has[gx, gy]]
        return float(v.mean()) if v.size >= 4 else None

    def is_pine(x, y):
        c = disc_mean(colour, colour_n > 0, x, y)
        f = disc_mean(canopy, high + opened > 0, x, y)
        return c is not None and f is not None and c >= a.pine and f >= a.canopy

    def top_at(x, y):
        gx, gy = disc(x, y, a.radius)
        v = chm0[gx, gy]
        return float(v.max()) if v.size else 0.0

    labels, count = ndimage.label((colour >= a.pine) & (canopy >= a.canopy), np.ones((3, 3)))
    patches, hosts = [], {}
    for k in range(1, count + 1):
        ix, iy = np.nonzero(labels == k)
        if len(ix) < a.min_cells:
            continue
        px, py = float(XS[ix].mean()), float(XS[iy].mean())
        if abs(px) > 30 or abs(py) > 30:
            continue
        if a.debug and math.hypot(px - a.debug[0], py - a.debug[1]) < 3:
            print("patch", round(px, 2), round(py, 2), len(ix), [(round(bx, 2), round(by, 2), round(math.hypot(px - bx, py - by), 2), is_pine(bx, by), disc_mean(colour, colour_n > 0, bx, by)) for bx, by in base if math.hypot(px - bx, py - by) < 3])
        near = [(math.hypot(px - bx, py - by), i) for i, (bx, by) in enumerate(base)
                if math.hypot(px - bx, py - by) <= a.exclude]
        if not near:
            patches.append((px, py, len(ix), None))
            continue
        if not a.attach or any(is_pine(*base[i]) for _, i in near):
            continue
        d, i = min(near)
        if top_at(*base[i]) >= a.host_max_top:
            continue
        if i not in hosts or hosts[i][2] < len(ix):
            hosts[i] = (px, py, len(ix), i)
    patches += list(hosts.values())

    truth = {n: (x, y) for n, x, y in parse_tree_truth(a.world)}
    removed = []
    if a.log:
        opener = gzip.open if a.log.endswith('.gz') else open
        mm = re.search(r'\[test\] removed \d+: (.*)', opener(a.log, 'rt', errors='replace').read())
        removed = [s.strip() for s in mm.group(1).split(',')]
    print(f'{len(patches) - len(hosts)} pine-coloured canopy patches with no baseline tree '
          f'within {a.exclude} m; {len(hosts)} attached to a non-pine baseline tree')
    if a.list_patches:
        for px, py, cells, host in patches:
            d, n = min((math.hypot(px - x, py - y), n) for n, (x, y) in truth.items())
            print(f'  patch ({px:.1f}, {py:.1f}) {cells} cells host={host}: nearest trunk {n} {d:.2f} m')
    if a.admit_ticks:
        recent = [p for tick in positions[-a.admit_ticks:] for p in tick]
        print(f'admission check: {len(positions)} pre-freeze ticks recorded')
        for px, py, cells, host in patches:
            if host is not None:
                continue
            hits = [(dx, dy) for dx, dy in recent if math.hypot(dx - px, dy - py) <= a.admit_dist]
            if hits:
                d, n = min((math.hypot(px - x, py - y), n) for n, (x, y) in truth.items())
                print(f'  admit patch ({px:.1f}, {py:.1f}) {cells} cells: {len(hits)} hits; '
                      f'nearest trunk {n} {d:.2f} m')
    pairs, used_t, used_b = {}, set(), set()        # identity: baseline index -> trunk
    cand = sorted((math.hypot(bx - tx, by - ty), i, n) for i, (bx, by) in enumerate(base)
                  for n, (tx, ty) in truth.items()
                  if math.hypot(bx - tx, by - ty) <= 2.0)
    for d, i, n in cand:
        if i not in used_b and n not in used_t:
            pairs[i] = n
            used_b.add(i)
            used_t.add(n)
    fired = []
    site_radius = a.site_radius or a.radius
    for px, py, cells, host in patches:
        gx, gy = disc(px, py, site_radius)
        visits, cur = [], []
        for t, h, op in after:
            hh, oo = h[gx, gy].sum(), op[gx, gy].sum()
            if hh + oo < 1:
                continue
            if cur and t - cur[-1][0] > a.gap:
                visits.append(cur)
                cur = []
            cur.append((t, hh / (hh + oo)))
        if cur:
            visits.append(cur)
        means = [(v[-1][0], float(np.mean([f for _, f in v]))) for v in visits]
        candidate = False
        if a.debug and math.hypot(px - a.debug[0], py - a.debug[1]) < 3:
            for tm, f in means:
                c1 = chm_at(tm)[gx, gy]
                c0 = chm0[gx, gy]
                print(f'  visit end +{tm - t_freeze_sim:.0f}s mean {f:.2f} top '
                      f'{c0[c0 >= 0].max() if (c0 >= 0).any() else -1:.1f} -> '
                      f'{c1[c1 >= 0].max() if (c1 >= 0).any() else -1:.1f}')
        for k in range(1, len(means)):
            if means[k][1] < a.open and means[k - 1][1] < a.open:
                chm1 = chm_at(means[k][0])
                top0 = chm0[gx, gy][chm0[gx, gy] >= 0]
                top1 = chm1[gx, gy][chm1[gx, gy] >= 0]
                if not (top0.size and top1.size):
                    drop = 0.0
                elif a.drop_mode == 'mean':
                    drop = top0.mean() - top1.mean()
                else:
                    drop = top0.max() - top1.max()
                if a.show_candidates and not candidate:
                    candidate = True
                    print(f'  candidate ({px:.1f}, {py:.1f}) host={host} +{means[k][0] - t_freeze_sim:.0f}s '
                          f'max drop {top0.max() - top1.max() if top0.size and top1.size else 0:.2f} '
                          f'mean drop {top0.mean() - top1.mean() if top0.size and top1.size else 0:.2f}')
                if drop >= a.drop - 1e-6:
                    fired.append((px, py, cells, means[k][0] - t_freeze_sim, drop, host))
                    break
    for px, py, cells, t, drop, host in fired:
        if host is not None:
            bx, by = base[host]
            paired = pairs.get(host)
            verdict = (f'credits {paired}' if paired in removed else
                       f'FALSE (paired with standing {paired})' if paired else
                       'unpaired baseline tree (phantom): FALSE')
            print(f'  LOST of baseline tree at ({bx:.1f}, {by:.1f}) from its patch '
                  f'({px:.1f}, {py:.1f}) {cells} cells, +{t:.0f} s, top drop {drop:.1f} m -> {verdict}')
            continue
        near = min(((math.hypot(px - truth[n][0], py - truth[n][1]), n) for n in removed),
                   default=(float('inf'), '-'))
        tree = min(((math.hypot(px - x, py - y), n) for n, (x, y) in truth.items()))
        verdict = ('credits ' + near[1] if near[0] <= 3.0 else
                   'FALSE (> 6 m from every cut)' if near[0] > 6.0 else 'near a cut, not credited')
        print(f'  alert at ({px:.1f}, {py:.1f}) {cells} cells, +{t:.0f} s, top drop {drop:.1f} m; '
              f'nearest trunk {tree[1]} {tree[0]:.1f} m -> {verdict}')
    print(f'alerts: {len(fired)}')


if __name__ == '__main__':
    main()
