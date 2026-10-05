"""Recompute LOST-gate rules on a LIVE removal-test bag.

Uses what the live tracker saw: its frozen baseline (/parrot_tree_baseline),
its detections every tick (/parrot_tree_positions), the live canopy map and
change map (incl. -30 height-drop cells). A baseline tree is unmatched when no
detection lies within min(0.4 x nearest-baseline-neighbour, 1.5 m) (the
tracker's association radius, ignoring one-to-one). Rules fire on 3
consecutive unmatched ticks.

usage: live_gate_eval.py BAG LAUNCH_LOG WORLD [BAG LOG WORLD ...]
"""
import math
import re
import sys

import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2

from deforestation_monitoring.removal_test import _cloud_xyz
from deforestation_monitoring.tree_detection import parse_tree_truth
from deforestation_monitoring.visibility import top_drop

W = '/home/alig/rs1_18_ws/src/RS1_18/41068_ignition_bringup/worlds/'
RULES = {
    'max top>=0.6 & g+d>=6 (current)': lambda v: v['max'] >= 0.6 and v['gd'] >= 6,
    'max top>=0.5 & g+d>=6': lambda v: v['max'] >= 0.5 and v['gd'] >= 6,
    'cells top>=0.6 & g+d>=6': lambda v: v['cells'] >= 0.6 and v['gd'] >= 6,
    'frac>=0.8 & g+d>=6': lambda v: v['frac'] >= 0.8 and v['gd'] >= 6,
    'frac>=0.9 & g+d>=6': lambda v: v['frac'] >= 0.9 and v['gd'] >= 6,
    'frac>=0.9 & max>=0.3 & g+d>=6': lambda v: v['frac'] >= 0.9 and v['max'] >= 0.3 and v['gd'] >= 6,
    'current & nearest det >= 1.5': lambda v: v['frac'] >= 0.9 and v['max'] >= 0.3 and v['gd'] >= 6 and v['nd'] >= 1.5,
    'current & nearest det >= 2.0': lambda v: v['frac'] >= 0.9 and v['max'] >= 0.3 and v['gd'] >= 6 and v['nd'] >= 2.0,
    'current & nearest det >= 2.5': lambda v: v['frac'] >= 0.9 and v['max'] >= 0.3 and v['gd'] >= 6 and v['nd'] >= 2.5,
}


def grid(m):
    g = np.array(m.data, np.float32).reshape(m.info.height, m.info.width).T
    return np.where(g >= 0, g / 10, -1)


def evaluate(bag, log, world):
    txt = open(log, errors='replace').read()
    t_rem = float(re.search(
        r'\[(\d+\.\d+)\] \[removal_test\]: \[test\] removed \d+: (.*)', txt).group(1))
    names = re.search(r'\[test\] removed \d+: (.*)', txt).group(1).split(', ')
    truth = parse_tree_truth(W + world + '.sdf')
    r = SequentialReader()
    r.open(StorageOptions(uri=bag, storage_id='sqlite3'),
           ConverterOptions('cdr', 'cdr'))
    base = freeze = cur = chg = None
    rows = {}
    while r.has_next():
        topic, data, ts = r.read_next()
        t = ts / 1e9
        if topic == '/parrot_tree_baseline':
            pts = _cloud_xyz(deserialize_message(data, PointCloud2))
            base = {int(round(z)): (float(x), float(y)) for x, y, z in pts}
            freeze = cur
            ids = list(base)
            xy = np.array([base[i] for i in ids])
            d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
            np.fill_diagonal(d, np.inf)
            radius = dict(zip(ids, np.minimum(0.4 * d.min(axis=1), 1.5)))
        elif topic == '/forest_canopy_map':
            cur = grid(deserialize_message(data, OccupancyGrid))
        elif topic == '/canopy_change_map':
            m = deserialize_message(data, OccupancyGrid)
            chg = np.array(m.data, np.int8).reshape(m.info.height, m.info.width)
        elif topic == '/parrot_tree_positions' and base is not None \
                and t >= t_rem and chg is not None and freeze is not None:
            pos = _cloud_xyz(deserialize_message(data, PointCloud2))
            rr, cc = np.nonzero(chg < 0)
            wx, wy = -40 + (cc + .5) * .25, -40 + (rr + .5) * .25
            for i, (x, y) in base.items():
                if len(pos) and np.min(np.hypot(pos[:, 0] - x, pos[:, 1] - y)) <= radius[i]:
                    continue
                gd = int(((wx - x) ** 2 + (wy - y) ** 2 <= 2.0 ** 2).sum())
                nd = (float(np.min(np.hypot(pos[:, 0] - x, pos[:, 1] - y)))
                      if len(pos) else 99.0)
                rows.setdefault(i, []).append(dict(
                    t=t, gd=gd, nd=nd,
                    max=top_drop(freeze, cur, .25, (-40, -40), x, y, 1.0, 'max'),
                    cells=top_drop(freeze, cur, .25, (-40, -40), x, y, 1.0, 'cells'),
                    frac=top_drop(freeze, cur, .25, (-40, -40), x, y, 1.0, 'frac')))
    # identity pairing (closest first, <= 2 m)
    cands = sorted((math.hypot(bx - tx, by - ty), i, n) for i, (bx, by) in base.items()
                   for n, tx, ty in truth if math.hypot(bx - tx, by - ty) <= 2.0)
    trunk, used = {}, set()
    for _, i, n in cands:
        if i not in trunk and n not in used:
            trunk[i] = n
            used.add(n)
    if DUMP:
        import json, os
        out = os.path.join(DUMP, os.path.basename(bag.rstrip('/')) + '_gate.json')
        json.dump({'names': names, 'trunk': {str(k): v for k, v in trunk.items()},
                   'rows': {str(k): v for k, v in rows.items()}}, open(out, 'w'))
    print(f'\n{bag}: {len(names)} removed, {sum(1 for n in names if n in used)} in the baseline')
    for rname, rule in RULES.items():
        caught, fs, ph = [], [], []
        for i, rs in rows.items():
            run, last, fired = 0, None, False
            for v in rs:
                if last is not None and v['t'] - last > 2.5:
                    run = 0
                last = v['t']
                run = run + 1 if rule(v) else 0
                if run >= 3:
                    fired = True
                    break
            if not fired:
                continue
            n = trunk.get(i)
            if n in names:
                caught.append(n)
            elif n is None:
                ph.append(f'#{i}')
            else:
                fs.append(n)
        missed = [n for n in names if n in used and n not in caught]
        if rname.startswith('max') or rname.startswith('cells') or rname.startswith('frac>=0.8'):
            continue
        print(f'  {rname:34s} caught {len(caught)}  standing false {fs}  phantom {ph}  missed(baselined) {missed}')


args = sys.argv[1:]
DUMP = None
if args and args[0].startswith('--dump='):
    DUMP = args.pop(0).split('=', 1)[1]
for k in range(0, len(args), 3):
    evaluate(*args[k:k + 3])
