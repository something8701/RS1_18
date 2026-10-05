#!/usr/bin/env python3
"""Replay a recorded removal test through the real mapper + tracker offline.

The bag's raw ``/parrot1/scan`` and ``/parrot1/odometry`` are fed to the
real ``ScanMapper`` and ``ParrotTreeTracker`` classes on bag time, with no
simulator and no DDS. If the bag has the drone camera (RGB + depth),
``CameraSpeciesMapper`` runs too. Every publish period the mapper's outputs
go straight to the tracker's callbacks, like the live topics. The trees were
already cut in the recording, so the result is scored with the same
``score_removal`` and visibility tiers as the live ``removal_test``.

    ros2 run deforestation_monitoring offline_replay \\
        --bag ~/rs1_18_ws/bags/removal33 --world dense_forest \\
        --log ~/rs1_18_ws/eval/logs/removal33.log \\
        [--min-loops 2] [--set tracker.lost_evidence_cells=8]

``--log`` gives the cut time and cut list; recordings flown with the
two-loop baseline need ``--min-loops 2``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import yaml

# Parameters of monitoring_nodes.launch.py (keep in sync).
MAPPER_PARAMS = {
    'resolution': 0.25, 'map_size_x': 80.0, 'map_size_y': 80.0,
    'scan_topic': '/parrot1/scan', 'odom_topic': '/parrot1/odometry',
    'map_frame': 'parrot1_odom', 'sensor_pitch': 1.5708, 'sensor_yaw': 0.0,
    'altitude': 10.0, 'canopy_threshold': 2.0, 'height_scale': 10.0,
    'baseline_threshold': 20, 'baseline_loop_scans': 15,
    'coverage_required': 0.9, 'publish_rate': 1.0, 'drop_evidence_m': 1.0,
    'baseline_fill': True,
}
TRACKER_PARAMS = {
    'height_scale': 10.0, 'min_baseline_cells': 150, 'min_baseline_trees': 2,
    'baseline_coverage': 0.9, 'track_radius': 0.6, 'lost_streak_threshold': 3,
    'lost_evidence_cells': 6, 'lost_drop_evidence': True,
    'lost_top_drop': 0.3, 'lost_top_frac': 0.9, 'lost_mean_drop': 2.0, 'terrain_fill': False,
    'publish_rate': 1.0, 'seen_lost': True, 'baseline_mean_chm': True,
    'merge_lost_crowns': True, 'camera_area_alerts': True, 'area_attach': True, 'freeze_median_ticks': 30, 'freeze_median_radius': 2.0, 'freeze_median_exclusive': True, 'merge_hold_s': 240.0, 'merge_after_ratio': 0.6,
}
# The tracker keeps trees out to ±37 m in the dense forest, its camera patches
# 7 m inside that (monitoring_nodes.launch.py); the coverage gate stays BOX.
TRACKER_BOX = {'dense_forest': (37.0, 7.0)}
TRACKER_WORLD: Dict[str, Dict] = {}   # per-world tracker params (launch expressions)
BOX = {'dense_forest': 30.0, 'sparse_trees': 15.0, 'cluster_test': 10.0,
       'showcase_forest': 30.0, 'simple_trees': 20.0}


class _Sink:
    """Stand-in publisher: records messages and forwards them."""

    def __init__(self, forward=None, keep=False):
        self.forward, self.keep, self.msgs, self.last = forward, keep, [], None

    def publish(self, msg):
        self.last = msg
        if self.keep:
            self.msgs.append(msg)
        if self.forward is not None:
            self.forward(msg)


def _coerce(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return {'true': True, 'false': False}.get(v.lower(), v)


def removal_time_from_log(path: str) -> Optional[float]:
    """Epoch of the ``[test] removed N:`` line in a launch log (.gz is fine)."""
    pat = re.compile(r'\[(\d+\.\d+)\] \[removal_test\]: \[test\] removed \d+')
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt', errors='replace') as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                return float(m.group(1))
    return None


def removed_names_from_log(path: str) -> List[str]:
    """Tree names of the ``[test] removed N: a, b`` line (the cut that was
    really made), or [] if the log has none."""
    pat = re.compile(r'\[test\] removed \d+: (.*)$')
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt', errors='replace') as fh:
        for line in fh:
            m = pat.search(line.rstrip())
            if m:
                return [n.strip() for n in m.group(1).split(',') if n.strip()]
    return []


def replay(bag: str, world: str, t_removed: Optional[float],
           overrides: Dict[str, Dict], n_trees: int = 10,
           balanced: bool = False, quiet: bool = True,
           progress: bool = True, canopy_only: bool = False,
           names: Optional[List[str]] = None) -> Dict:
    import rclpy
    from rclpy.logging import LoggingSeverity, set_logger_level
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.serialization import deserialize_message
    from rclpy.time import Time
    from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Image, LaserScan
    from ament_index_python.packages import get_package_share_directory

    from . import camera_species_mapper as cam_mod
    from . import scan_mapper as sm_mod
    from . import parrot_tree_tracker as tr_mod
    from .removal_test import (_cloud_xyz, baseline_quality, format_report,
                               parse_canopy_events, score_removal,
                               select_removal_targets)
    from .tree_detection import parse_tree_truth
    from .visibility import classify_tree, disc_max, removal_tier, top_drop

    share = get_package_share_directory('deforestation_monitoring')
    det_yaml = yaml.safe_load(open(os.path.join(
        share, 'config', 'tree_detection_params.yaml')))
    half = BOX.get(world, 30.0)
    box = {'x_min': -half, 'x_max': half, 'y_min': -half, 'y_max': half}
    mapper_p = dict(MAPPER_PARAMS, use_sim_time=True,
                    **{f'coverage_{k}': v for k, v in box.items()})
    t_half, t_margin = TRACKER_BOX.get(world, (half, 0.0))
    tracker_p = dict(det_yaml['/**']['ros__parameters'], **TRACKER_PARAMS,
                     use_sim_time=True, area_edge_margin=t_margin,
                     survey_x_min=-t_half, survey_x_max=t_half,
                     survey_y_min=-t_half, survey_y_max=t_half)
    tracker_p.update(TRACKER_WORLD.get(world, {}))
    mapper_p.update(overrides.get('mapper', {}))
    tracker_p.update(overrides.get('tracker', {}))
    params = {'scan_mapper': mapper_p, 'parrot_tree_tracker': tracker_p,
              'camera_species_mapper': dict(use_sim_time=True,
                                            **overrides.get('camera', {}))}

    orig_init = Node.__init__

    def init(self, node_name, *a, **kw):
        extra = [Parameter(k, value=v)
                 for k, v in params.get(node_name, {}).items()]
        kw['parameter_overrides'] = extra + list(
            kw.get('parameter_overrides') or [])
        orig_init(self, node_name, *a, **kw)

    Node.__init__ = init
    if not rclpy.ok():
        rclpy.init(args=['--ros-args', '--log-level', 'warn'] if quiet else None)
    try:
        mapper = sm_mod.ScanMapper()
        tracker = tr_mod.ParrotTreeTracker()
        camera = cam_mod.CameraSpeciesMapper()
    finally:
        Node.__init__ = orig_init
    if quiet:
        for n in (mapper, tracker, camera):
            set_logger_level(n.get_logger().name, LoggingSeverity.WARN)

    # Wire mapper outputs to tracker inputs; record what the test node saw.
    canopy_events: List = []
    sink_map = _Sink(tracker.canopy_cb)
    mapper.canopy_pub = sink_map
    mapper.hits_pub = _Sink(tracker.hits_cb)
    mapper.change_pub = _Sink(tracker.change_cb)
    mapper.coverage_pub = _Sink(tracker.coverage_cb)
    mapper.baseline_pub = _Sink(tracker.baseline_cb)
    mapper.change_events_pub = _Sink(lambda m: canopy_events.append(
        (now_s[0], m.data)))
    for name in ('change_marker_pub', 'swath_pub'):
        setattr(mapper, name, _Sink())
    events: List[Dict] = []
    tracker.event_pub = _Sink(lambda m: events.append({
        'type': m.event_type, 'id': int(m.tree_id), 'x': float(m.x),
        'y': float(m.y), 't_epoch': now_s[0]}))
    base_sink = _Sink()
    pos_sink = _Sink()
    tracker.baseline_pub = base_sink
    tracker.positions_pub = pos_sink
    for name in ('status_pub', 'marker_pub', 'treetop_pub', 'crown_pub',
                 'drone_baseline_pub'):
        setattr(tracker, name, _Sink())
    tracker.canopy_alert_pub = _Sink(lambda m: canopy_events.append((now_s[0], m.data)))
    info_sink = _Sink()                 # latest /parrot_tree_baseline_info
    notes: List[Dict] = []              # /parrot_tree_change_notes (LOST by, MERGED)
    tracker.baseline_info_pub = info_sink
    tracker.change_note_pub = _Sink(lambda m: notes.append(
        dict(json.loads(m.data), t_epoch=now_s[0])))
    camera.species_pub = _Sink(tracker.species_map_cb)
    camera.seen_map_pub = _Sink(tracker.seen_map_cb)
    camera.seen_pub = _Sink(tracker.seen_recent_cb)
    frames: Dict[tuple, Dict] = {}    # header stamp -> {'rgb': msg, 'depth': msg}

    # LOST-gate diagnostics: _lost_cells_near is only called for baseline
    # trees that are unmatched this tick, so this records, per tree id,
    # every unmatched tick and its lost-evidence cell count.
    gate_log: Dict[int, List] = {}
    freeze_map: List = [None]
    freeze_ring: List = []      # tracker positions, last 60 ticks up to freeze+3
    orig_near = tracker._lost_cells_near

    from scipy import ndimage
    cache: Dict = {}

    def features(name, mask):
        # (#lost 8-neighbours, component labels) of a mask, once per tick
        key = (name, now_s[0])
        if cache.get(name, (None,))[0] != key:
            m = mask.astype(np.int16)
            nb = ndimage.convolve(m, np.ones((3, 3), np.int16),
                                  mode='constant') - m
            lab, _ = ndimage.label(mask, structure=np.ones((3, 3)))
            cache[name] = (key, (nb, lab))
        return cache[name][1]

    def metrics(name, mask, tree, radius):
        if mask is None or not np.any(mask):
            return 0, 0, 0
        nb, lab = features(name, mask)
        rows, cols = np.nonzero(mask)
        wx = tracker.change_ox + (cols + 0.5) * tracker.change_res
        wy = tracker.change_oy + (rows + 0.5) * tracker.change_res
        close = (wx - tree['x']) ** 2 + (wy - tree['y']) ** 2 <= radius ** 2
        r, c = rows[close], cols[close]
        if r.size == 0:
            return 0, 0, 0
        clustered = int((nb[r, c] >= 1).sum())
        comp = int(np.bincount(lab[r, c]).max())
        return int(r.size), clustered, comp

    def spy_near(tree, radius):
        count = orig_near(tree, radius)
        ground = tracker.change_lost
        drop = getattr(tracker, 'change_dropped', None)
        both = ground
        if ground is not None and drop is not None:
            key = ('both', now_s[0])
            if cache.get('bothmask', (None,))[0] != key:
                cache['bothmask'] = (key, ground | drop)
            both = cache['bothmask'][1]
        g = metrics('ground', ground, tree, radius)
        b = metrics('both', both, tree, radius)
        # Own-top drop: max canopy within 0.75 / 1.0 m of the baseline
        # position, map at freeze vs latest map.
        tops = []
        if freeze_map[0] is not None and sink_map.last is not None:
            key = ('now', now_s[0])
            if cache.get('nowmap', (None,))[0] != key:
                cache['nowmap'] = (key, grid_of(sink_map.last))
            cur = cache['nowmap'][1]
            for rad in (0.75, 1.0):
                a = disc_max(*freeze_map[0], tree['x'], tree['y'], rad)
                c = disc_max(*cur, tree['x'], tree['y'], rad)
                tops.append(round(a - c, 2) if a is not None and c is not None
                            else 0.0)
            for rad in (0.75, 1.0):
                tops.append(round(top_drop(
                    freeze_map[0][0], cur[0], cur[1], cur[2],
                    tree['x'], tree['y'], rad, 'cells'), 2))
            tops.append(round(top_drop(
                freeze_map[0][0], cur[0], cur[1], cur[2],
                tree['x'], tree['y'], 1.0, 'frac'), 2))
        else:
            tops = [0.0, 0.0, 0.0, 0.0, 0.0]
        # distance to the nearest detection of the previous tick
        near_det = 99.0
        if pos_sink.last is not None:
            key = ('pos', now_s[0])
            if cache.get('pos', (None,))[0] != key:
                cache['pos'] = (key, _cloud_xyz(pos_sink.last))
            pts = cache['pos'][1]
            if len(pts):
                near_det = float(np.min(np.hypot(pts[:, 0] - tree['x'],
                                                 pts[:, 1] - tree['y'])))
        gate_log.setdefault(tree['id'], []).append(
            (now_s[0], count, g[0], b[0] - g[0], g[1], g[2], b[1], b[2],
             tops[0], tops[1], tops[2], tops[3], tops[4],
             round(near_det, 2)))
        return count
    tracker._lost_cells_near = spy_near

    def grid_of(msg):
        g = np.array(msg.data, dtype=np.float32).reshape(
            msg.info.height, msg.info.width).T
        return (np.where(g >= 0, g / 10.0, -1.0), msg.info.resolution,
                (msg.info.origin.position.x, msg.info.origin.position.y))

    reader = SequentialReader()
    reader.open(StorageOptions(uri=bag, storage_id='sqlite3'),
                ConverterOptions('cdr', 'cdr'))
    now_s = [0.0]
    next_tick = None
    pre_map = None
    t_freeze = None
    coverage_at_removal = None
    clocks = (mapper.get_clock(), tracker.get_clock())
    camera_topics = {'/parrot1/camera/image': 'rgb', '/parrot1/camera/depth/image': 'depth'}
    t0 = time.time()
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        if topic in camera_topics:
            # Camera stamps are sim time; the camera node's clock follows the
            # odometry stamps (below) so its recent-window test is consistent.
            msg = deserialize_message(data, Image)
            key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
            pair = frames.setdefault(key, {})
            pair[camera_topics[topic]] = msg
            if len(pair) == 2:
                camera.frame_cb(pair['rgb'], pair['depth'])
                del frames[key]
            continue
        if topic == '/survey_status':
            # loop counting for baseline_min_loops / freeze_min_loops
            from std_msgs.msg import String as _String
            status = deserialize_message(data, _String)
            if mapper.baseline_min_loops > 0:
                mapper.survey_status_callback(status)
            if tracker.freeze_min_loops > 0:
                tracker.survey_cb(status)
            continue
        if topic not in ('/parrot1/scan', '/parrot1/odometry'):
            continue
        now_s[0] = stamp / 1e9
        for c in clocks:
            c.set_ros_time_override(Time(nanoseconds=stamp))
        if topic == '/parrot1/odometry':
            odom = deserialize_message(data, Odometry)
            mapper.odom_callback(odom)
            camera.odom_cb(odom)
            camera.get_clock().set_ros_time_override(Time.from_msg(odom.header.stamp))
        else:
            mapper.scan_callback(deserialize_message(data, LaserScan))
        if next_tick is None:
            next_tick = now_s[0] + 1.0
            t_first = now_s[0]
        while now_s[0] >= next_tick:
            next_tick += 1.0
            if progress and int(next_tick) % 100 == 0:
                print(f'  [{os.path.basename(bag)}] bag t={next_tick - t_first:.0f}s '
                      f'coverage {tracker.coverage_pct * 100:.0f}% baseline '
                      f'{tracker.baseline_tree_count if base_sink.last else "-"}'
                      f' events {len(events)}  ({time.time() - t0:.0f}s wall)',
                      file=sys.stderr, flush=True)
            mapper.publish_maps()
            camera.publish()
            tracker.publish_status()
            if pos_sink.last is not None and (
                    t_freeze is None or now_s[0] <= t_freeze + 3):
                freeze_ring.append((now_s[0], [
                    (round(float(x), 2), round(float(y), 2))
                    for x, y, _ in _cloud_xyz(pos_sink.last)]))
                del freeze_ring[:-60]
            if t_freeze is None and base_sink.last is not None:
                t_freeze = now_s[0]
                freeze_map[0] = grid_of(sink_map.last)
            if (t_removed is not None and pre_map is None
                    and now_s[0] >= t_removed and sink_map.last is not None):
                pre_map = grid_of(sink_map.last)
                coverage_at_removal = tracker.coverage_pct
    wall = time.time() - t0

    sdf = os.path.join(get_package_share_directory('41068_ignition_bringup'),
                       'worlds', f'{world}.sdf')
    truth = parse_tree_truth(sdf)
    box_t = (-half, half, -half, half)
    baseline = ([(int(round(z)), float(x), float(y))
                 for x, y, z in _cloud_xyz(base_sink.last)]
                if base_sink.last is not None else [])
    final_map = grid_of(sink_map.last) if sink_map.last is not None else None
    detections = ([(float(x), float(y)) for x, y, _ in
                   _cloud_xyz(pos_sink.last)] if pos_sink.last else [])
    out = {'bag': bag, 'world': world, 'replay_s': wall,
           't_freeze': t_freeze, 't_removed': t_removed,
           'baseline': baseline, 'events': events,
           'baseline_info': (json.loads(info_sink.last.data) if info_sink.last else None),
           'notes': notes,
           'freeze_positions': freeze_ring,
           'quality': baseline_quality(truth, baseline, box_t)}
    if t_removed is None:
        return out
    # n_trees 0 = a no-change recording scored as a guard (nothing removed).
    if names:      # an explicit cut list (removal_test trees:=a,b,c)
        removed = [t for t in truth if t[0] in names]
    else:
        removed = (select_removal_targets(truth, n_trees, box_t, balanced=balanced,
                                          canopy_only=canopy_only)
                   if n_trees > 0 else [])
    tiers = {}
    for tree in removed:
        cls, top0 = (classify_tree(*pre_map, tree) if pre_map is not None
                     else ('unobserved', None))
        cls = removal_tier(tree, cls, truth)
        top1 = (disc_max(*final_map, tree[1], tree[2], 0.75)
                if final_map is not None else None)
        tiers[tree[0]] = {'class': cls, 'top_before': top0, 'top_after': top1}
    after = [dict(e, t=e['t_epoch'] - t_removed, before_removal=False)
             for e in events if e['t_epoch'] >= t_removed]
    before = [e for e in events if e['t_epoch'] < t_removed]
    canopy = [c for ts, txt in canopy_events if ts >= t_removed
              for c in parse_canopy_events(txt)]
    area = [(c[1], c[2]) for c in canopy if c[0] == 'LOST']
    result = score_removal(removed, truth, baseline, after, canopy,
                           detections, 1.5, tiers=tiers, area_alerts=area)
    for e in before:
        result['false_event_list'].append(
            {**e, 'nearest_tree': None, 'nearest_dist': float('nan'),
             'nearest_was_removed': False,
             'why': 'event before any tree was removed'})
    result['false_events'] = len(result['false_event_list'])
    result['passed'] = result['passed'] and not before
    # Gate margin: removed (paired) vs standing baseline trees, after removal.
    removed_ids = {t['baseline_id'] for t in result['trees']
                   if t.get('baseline_id') is not None}
    gate = {}
    for tid, rows in gate_log.items():
        post = [r for r in rows if r[0] >= t_removed]
        mx = lambda i: max((r[i] for r in post), default=0)  # noqa: E731
        gate[tid] = {'unmatched_ticks': len(post),
                     'max_evidence': mx(1), 'max_ground': mx(2),
                     'max_drop': mx(3),
                     'max_both': max((r[2] + r[3] for r in post), default=0),
                     'ground_clustered': mx(4), 'ground_comp': mx(5),
                     'both_clustered': mx(6), 'both_comp': mx(7),
                     'top_drop075': mx(8), 'top_drop100': mx(9),
                     'rows': post,
                     'removed': tid in removed_ids}
    out['lost_gate'] = gate
    q = out['quality']
    meta = {
        'world': world, 'mode': f'OFFLINE replay of {os.path.basename(bag)}',
        'coverage at removal': (f'{coverage_at_removal * 100:.1f}%'
                                if coverage_at_removal is not None else '?'),
        'baseline vs SDF (1 m)': (
            f"{q['baseline']} trees, TP {q['tp']}, FP {q['fp']}, "
            f"FN {q['fn']} of {q['truth_in_box']} in the survey box"),
        'observed after removal': f'{now_s[0] - t_removed:.0f} s',
    }
    out.update(result=result, report=format_report(result, meta),
               removed=removed)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--bag', required=True)
    ap.add_argument('--world', required=True)
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--t-removed', type=float,
                   help='epoch seconds of the removal')
    g.add_argument('--log', help='launch log with the "[test] removed" line')
    g.add_argument('--report-json', help='removal_test JSON (t_removed_epoch)')
    ap.add_argument('--n-trees', type=int, default=10)
    ap.add_argument('--balanced', action='store_true')
    ap.add_argument('--canopy-only', action='store_true')
    ap.add_argument('--min-loops', type=int, default=0,
                    help='survey loops before the baselines freeze (launch: 2 in '
                         'the dense world since 2026-10-05; recordings before that '
                         'were cut after 1, so 0 = the coverage rule only)')
    ap.add_argument('--trees', default='',
                    help='comma-separated cut list (as removal_test trees:=...)')
    ap.add_argument('--set', action='append', default=[],
                    metavar='NODE.PARAM=VALUE',
                    help='override, NODE is mapper, tracker or camera (repeatable)')
    ap.add_argument('--out', help='write the report here (.md) + .json')
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args(argv)

    t_removed = a.t_removed
    names = [n for n in a.trees.split(',') if n]
    if a.log:
        t_removed = removal_time_from_log(a.log)
        # score what the recording really cut; target selection has changed
        # since some recordings were made (showcase replays, 2026-10-05)
        names = names or removed_names_from_log(a.log)
    if a.report_json:
        t_removed = json.load(open(a.report_json)).get('t_removed_epoch')
    overrides: Dict[str, Dict] = {'mapper': {}, 'tracker': {}, 'camera': {}}
    if a.min_loops:
        overrides['mapper']['baseline_min_loops'] = a.min_loops
        overrides['tracker']['freeze_min_loops'] = a.min_loops
    for s in a.set:
        key, val = s.split('=', 1)
        node, param = key.split('.', 1)
        overrides[node][param] = _coerce(val)

    res = replay(os.path.expanduser(a.bag), a.world, t_removed, overrides,
                 a.n_trees, a.balanced, quiet=not a.verbose,
                 canopy_only=a.canopy_only,
                 names=names)
    print(f"replayed in {res['replay_s']:.0f} s; baseline {len(res['baseline'])}"
          f" trees; events {len(res['events'])}")
    if 'report' in res:
        print(res['report'])
    if res.get('lost_gate') is not None:
        g = res['lost_gate']
        by_id = {t['baseline_id']: t['name'] for t in res['result']['trees']
                 if t.get('baseline_id') is not None}
        print('LOST gate after removal (unmatched ticks, max lost-evidence '
              'cells while unmatched):')
        for tid, name in sorted(by_id.items(), key=lambda kv: kv[1]):
            v = g.get(tid, {'unmatched_ticks': 0, 'max_evidence': 0,
                            'max_ground': 0, 'max_drop': 0, 'max_both': 0})
            print(f'  removed  #{tid:<4d} {name:9s} unmatched {v["unmatched_ticks"]:4d}'
                  f'  evidence {v["max_evidence"]:3d} (ground {v["max_ground"]}, '
                  f'drop {v["max_drop"]}, ground+drop {v["max_both"]})')
        standing = sorted(((v['max_both'], v['max_ground'], v['max_drop'],
                            v['unmatched_ticks'], tid)
                           for tid, v in g.items() if not v['removed']),
                          reverse=True)[:8]
        for both, gr, dr, ticks, tid in standing:
            print(f'  standing #{tid:<4d} unmatched {ticks:4d}  ground {gr}, '
                  f'drop {dr}, ground+drop {both}')
    if a.out:
        with open(a.out, 'w') as fh:
            fh.write(res.get('report', ''))
        with open(os.path.splitext(a.out)[0] + '.json', 'w') as fh:
            json.dump({k: v for k, v in res.items() if k != 'report'}, fh,
                      indent=2, default=str)
    return 0 if res.get('result', {}).get('passed') else 1


if __name__ == '__main__':
    sys.exit(main())
