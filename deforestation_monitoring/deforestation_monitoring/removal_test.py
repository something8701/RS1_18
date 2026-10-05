#!/usr/bin/env python3
"""Live tree-removal acceptance test.

Scenario (run against a live simulation, see ``removal_test.launch.py``):

  1. wait until the drone survey reaches ``min_coverage`` (default 90%) and
     ``parrot_tree_tracker`` has frozen its individual-tree baseline;
  2. delete ``n_trees`` trees from the Gazebo world (default 10, chosen
     automatically: inside the survey box, well separated);
  3. keep patrolling for ``settle_loops`` survey-loop completions;
  4. score what the pipeline reported against the known removals:

     * right:  a LOST event at a removed tree (true positive);
     * missed: a removed tree with no LOST event (false negative), with a
               diagnosis of why;
     * wrong:  a LOST event at a tree that is still standing, any GAINED
               event, or an area alert far from every cut (false positives).

A markdown + JSON report is written to ``report_dir`` and the one-line
verdict is published on ``/removal_test_result``. PASS = every removed tree
found at its tier (see ``score_removal``) and zero false events.

The scoring functions are pure so they are unit-tested offline.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from deforestation_monitoring.survey_loops import LoopCounter  # noqa: F401 (re-exported)

Tree = Tuple[str, float, float]            # (name, x, y)
Point2 = Tuple[float, float]


# ── Pure logic ──────────────────────────────────────────────────────

def occluded_from_above(truth: Sequence[Tree],
                        radius: float = 7.0) -> set:
    """Pines within ``radius`` of an oak trunk (possibly under its crown).

    Optional, not used by default.
    """
    oaks = [t for t in truth if t[0].startswith('oak')]
    return {t[0] for t in truth if t[0].startswith('pine') and any(
        math.hypot(t[1] - o[1], t[2] - o[2]) <= radius for o in oaks)}


def select_removal_targets(
    truth: Sequence[Tree],
    n: int,
    box: Tuple[float, float, float, float],
    margin: float = 5.0,
    min_separation: float = 10.0,
    visible_only: bool = False,
    balanced: bool = False,
    canopy_only: bool = False,
) -> List[Tree]:
    """Pick ``n`` trees inside ``box`` (x_min, x_max, y_min, y_max) shrunk
    by ``margin``, most isolated first, at least ``min_separation`` apart so
    each removal is scored on its own, skipping trees hidden under another
    crown when ``visible_only``. ``canopy_only`` keeps only trees that are
    canopy tier by geometry (``visibility.removal_tier``): oaks, and pines
    with no oak trunk within ``CROWN_SHARE_RADIUS``. Deterministic."""
    from .visibility import CROWN_SHARE_RADIUS
    x0, x1, y0, y1 = box
    hidden = occluded_from_above(truth) if visible_only else set()
    if canopy_only:
        hidden |= occluded_from_above(truth, CROWN_SHARE_RADIUS)
    inside = [t for t in truth
              if x0 + margin <= t[1] <= x1 - margin
              and y0 + margin <= t[2] <= y1 - margin
              and t[0] not in hidden]

    def nn(t):
        return min((math.hypot(t[1] - o[1], t[2] - o[2])
                    for o in truth if o[0] != t[0]), default=float('inf'))

    ranked = sorted(inside, key=lambda t: (-nn(t), t[0]))
    if balanced:
        # Alternate species (oak, pine, oak, ...) so a demo removes both
        # kinds; within a species the most isolated come first.
        by_sp: Dict[str, List[Tree]] = {}
        for t in ranked:
            by_sp.setdefault(t[0].split('_')[0], []).append(t)
        order: List[Tree] = []
        queues = [by_sp[k] for k in sorted(by_sp)]
        while any(queues):
            for q in queues:
                if q:
                    order.append(q.pop(0))
        ranked = order
    chosen: List[Tree] = []
    for t in ranked:
        if all(math.hypot(t[1] - c[1], t[2] - c[2]) >= min_separation
               for c in chosen):
            chosen.append(t)
        if len(chosen) == n:
            break
    return chosen


def parse_canopy_events(text: str) -> List[Tuple[str, float, float]]:
    """``/canopy_change_events`` strings -> [(LOST|NEW, x, y)]."""
    out = []
    for part in text.split(';'):
        m = re.search(r'(CANOPY LOST|NEW CANOPY).*near \(([-0-9.]+), ([-0-9.]+)\)',
                      part)
        if m:
            kind = 'LOST' if m.group(1) == 'CANOPY LOST' else 'NEW'
            out.append((kind, float(m.group(2)), float(m.group(3))))
    return out


def _nearest(p: Point2, pts: Sequence[Point2]) -> Tuple[Optional[int], float]:
    best, best_d = None, float('inf')
    for i, q in enumerate(pts):
        d = math.hypot(p[0] - q[0], p[1] - q[1])
        if d < best_d:
            best, best_d = i, d
    return best, best_d


def score_removal(
    removed: Sequence[Tree],
    truth: Sequence[Tree],
    baseline: Sequence[Tuple[int, float, float]],
    events: Sequence[Dict],
    canopy_events: Sequence[Tuple[str, float, float]] = (),
    final_detections: Sequence[Point2] = (),
    match_radius: float = 1.5,
    baseline_radius: float = 1.0,
    canopy_radius: float = 5.0,
    tiers: Optional[Dict[str, Dict]] = None,
    area_alerts: Sequence[Point2] = (),
    area_radius: float = 3.0,
    false_area_radius: float = 6.0,
    unchanged_tol: float = 1.0,
    identity_radius: float = 2.0,
) -> Dict:
    """Score tracker events against the known removals.

    With ``tiers`` (``{name: {'class': visible|understory|unobserved,
    'top_before': m, 'top_after': m}}`` from ``visibility.classify_tree`` on
    the pre-removal canopy map):

    * canopy trees (visible) need a tree-level LOST;
    * understory trees pass with an area alert (``area_alerts``: scan_mapper
      CANOPY LOST clusters / pattern_scanner CLEARING flags) within
      ``area_radius``, or are "not observable" when the canopy over the trunk
      dropped by no more than ``unchanged_tol`` (1.0 m, the noise of that
      quantity over standing understory trunks);
    * false events are strict for both: any LOST at a standing tree, any
      GAINED, any area alert farther than ``false_area_radius`` from every
      removed trunk.

    Without ``tiers`` every tree is treated as canopy.

    Identity matching (``identity_radius`` > 0): baseline trees are paired
    one-to-one with SDF trunks (closest pairs first, within
    ``identity_radius``). A LOST whose baseline tree is paired with a removed
    trunk counts for that tree; one paired with a standing trunk is false,
    even if it lies near a removed tree. (Dense oaks are detected about 1 m
    off their trunk, so position alone would credit a neighbour's LOST.)
    Events from unpaired baseline trees fall back to position.

    ``baseline``: [(id, x, y)] frozen baseline trees.
    ``events``: [{'type': 'LOST'|'GAINED', 'id', 'x', 'y', 't'}] with t =
    seconds after the removal.
    """
    base_pts = [(b[1], b[2]) for b in baseline]
    lost = [e for e in events if e['type'] == 'LOST']
    gained = [e for e in events if e['type'] == 'GAINED']

    removed_names = {r[0] for r in removed}
    rindex = {r[0]: i for i, r in enumerate(removed)}

    # Identity: baseline tree <-> SDF trunk, one-to-one, closest first.
    trunk_of: Dict[int, str] = {}
    base_of: Dict[str, Tuple[int, float]] = {}
    if identity_radius > 0:
        cands = sorted(
            (math.hypot(b[1] - t[1], b[2] - t[2]), b[0], t[0])
            for b in baseline for t in truth
            if math.hypot(b[1] - t[1], b[2] - t[2]) <= identity_radius)
        for d, bid, tn in cands:
            if bid not in trunk_of and tn not in base_of:
                trunk_of[bid] = tn
                base_of[tn] = (bid, d)

    tree_event: Dict[int, int] = {}
    used = set()
    for ei, e in enumerate(lost):                 # 1. by identity
        tn = trunk_of.get(e['id'])
        if tn in removed_names and rindex[tn] not in tree_event:
            tree_event[rindex[tn]] = ei
            used.add(ei)
    pairs = []                                    # 2. unpaired: by position
    for ri, (_, rx, ry) in enumerate(removed):
        if ri in tree_event:
            continue
        for ei, e in enumerate(lost):
            if ei in used or e['id'] in trunk_of:
                continue
            d = math.hypot(e['x'] - rx, e['y'] - ry)
            if d <= match_radius:
                pairs.append((d, ri, ei))
    pairs.sort()
    for d, ri, ei in pairs:
        if ri in tree_event or ei in used:
            continue
        tree_event[ri] = ei
        used.add(ei)

    trees = []
    for ri, (name, rx, ry) in enumerate(removed):
        bi, bd = _nearest((rx, ry), base_pts)
        if identity_radius > 0:
            in_baseline = name in base_of
            bid = base_of[name][0] if in_baseline else None
            bdist = base_of[name][1] if in_baseline else bd
        else:
            in_baseline = bi is not None and bd <= baseline_radius
            bid = baseline[bi][0] if in_baseline else None
            bdist = bd
        entry = {
            'name': name, 'x': rx, 'y': ry,
            'in_baseline': in_baseline,
            'baseline_id': bid,
            'baseline_dist': bdist if bi is not None else None,
            'detected': ri in tree_event,
        }
        if ri in tree_event:
            e = lost[tree_event[ri]]
            entry.update(event_id=e['id'], latency_s=e['t'],
                         position_error=math.hypot(e['x'] - rx, e['y'] - ry))
        else:
            canopy_near = [c for c in canopy_events if c[0] == 'LOST'
                           and math.hypot(c[1] - rx, c[2] - ry) <= canopy_radius]
            fi, fd = _nearest((rx, ry), list(final_detections))
            still = fi is not None and fd <= baseline_radius
            entry['canopy_loss_seen'] = bool(canopy_near)
            entry['still_detected_dist'] = fd if fi is not None else None
            if not in_baseline:
                why = ('not in the tracker baseline as its own tree '
                       f'(nearest baseline tree {bd:.2f} m away) — '
                       'nothing to lose')
            elif still:
                why = (f'a detection is still at the tree ({fd:.2f} m): '
                       'a neighbouring crown now covers the spot, or the '
                       'removal did not take effect')
            elif not canopy_near:
                why = ('scan_mapper reported no canopy loss near it, so '
                       'the LOST evidence gate never opened')
            else:
                why = ('canopy loss was seen but the tracker never '
                       'declared it LOST (streak / evidence gate)')
            entry['why_missed'] = why
        trees.append(entry)

    false_events = []
    for ei, e in enumerate(lost):
        if ei in used:
            continue
        ti, td = _nearest((e['x'], e['y']), [(t[1], t[2]) for t in truth])
        near = truth[ti][0] if ti is not None else None
        paired = trunk_of.get(e['id'])
        if paired is not None and paired not in removed_names:
            why = (f'LOST for a tree that is still standing (baseline tree '
                   f'#{e["id"]} is {paired})')
        elif near in removed_names and td <= 3.0:
            why = 'duplicate / off-position LOST for a removed tree'
        elif paired is None and identity_radius > 0 and near in removed_names \
                and td <= 5.5:
            # A baseline crown with no trunk under it (a crown fragment): still
            # false, one removal reported as two trees.
            why = (f'double count: baseline #{e["id"]} was a phantom crown '
                   f'(no trunk within {identity_radius:.0f} m) inside '
                   f'removed {near}\'s crown reach')
        elif paired is None and identity_radius > 0:
            why = ('LOST for a phantom baseline crown (no trunk within '
                   f'{identity_radius:.0f} m) where nothing was removed')
        else:
            why = 'LOST for a tree that is still standing'
        false_events.append({
            **e, 'nearest_tree': paired or near, 'nearest_dist': td,
            'nearest_was_removed': (paired or near) in removed_names,
            'why': why,
        })
    for e in gained:
        ti, td = _nearest((e['x'], e['y']), [(t[1], t[2]) for t in truth])
        false_events.append({
            **e, 'nearest_tree': truth[ti][0] if ti is not None else None,
            'nearest_dist': td, 'nearest_was_removed': False,
            'why': 'GAINED — nothing was added to the world',
        })
    # An area alert (canopy-loss cluster / clearing flag)
    # farther than false_area_radius from every removed trunk is false too.
    seen: List[Point2] = []
    for a in area_alerts:
        if any(math.hypot(a[0] - b[0], a[1] - b[1]) < 1.0 for b in seen):
            continue
        seen.append(a)
        ri, rd = _nearest(a, [(r[1], r[2]) for r in removed])
        if rd is not None and rd <= false_area_radius:
            continue
        false_events.append({
            'type': 'AREA', 'id': -1, 'x': a[0], 'y': a[1], 't': float('nan'),
            'nearest_tree': removed[ri][0] if ri is not None else None,
            'nearest_dist': rd if rd is not None else float('nan'),
            'nearest_was_removed': True,
            'why': (f'canopy-loss alert > {false_area_radius:.0f} m from every '
                    'removed tree'),
        })

    for t in trees:
        info = (tiers or {}).get(t['name'], {})
        cls = info.get('class', 'canopy')
        t['tier'] = cls if cls in ('understory', 'unobserved',
                                   'crown-shared') else 'canopy'
        t['top_before'] = info.get('top_before')
        t['top_after'] = info.get('top_after')
        alert = any(math.hypot(a[0] - t['x'], a[1] - t['y']) <= area_radius
                    for a in area_alerts)
        t['area_alert'] = alert
        if t['tier'] == 'canopy':
            t['tier_ok'] = t['detected']
        elif t['tier'] == 'crown-shared':    # inside an oak crown footprint
            t['tier_ok'] = t['detected'] or alert
        elif t['tier'] == 'understory':
            unchanged = (t['top_before'] is not None and t['top_after'] is not None
                         and t['top_before'] - t['top_after'] <= unchanged_tol)
            t['not_observable'] = unchanged and not alert
            t['tier_ok'] = t['detected'] or alert or unchanged
        else:                       # never scanned before removal: not scored
            t['tier_ok'] = True
    canopy = [t for t in trees if t['tier'] == 'canopy']
    under = [t for t in trees if t['tier'] == 'understory']
    shared = [t for t in trees if t['tier'] == 'crown-shared']
    tp = sum(1 for t in trees if t['detected'])
    fn = len(trees) - tp
    fp = len(false_events)
    return {
        'removed': len(removed),
        'true_positives': tp,
        'missed': fn,
        'false_events': fp,
        'canopy_trees': len(canopy),
        'canopy_found': sum(1 for t in canopy if t['detected']),
        'crown_shared_trees': len(shared),
        'crown_shared_ok': sum(1 for t in shared if t['tier_ok']),
        'understory_trees': len(under),
        'understory_ok': sum(1 for t in under if t['tier_ok']),
        'passed': (len(removed) > 0 and fp == 0
                   and all(t['tier_ok'] for t in trees)),
        'trees': trees,
        'false_event_list': false_events,
        'canopy_new_events': sum(1 for c in canopy_events if c[0] == 'NEW'),
        'canopy_lost_events': sum(1 for c in canopy_events if c[0] == 'LOST'),
    }


def baseline_quality(truth: Sequence[Tree],
                     baseline: Sequence[Tuple[int, float, float]],
                     box: Tuple[float, float, float, float],
                     radius: float = 1.0) -> Dict:
    """Baseline vs SDF truth inside the survey box (one-to-one, greedy)."""
    x0, x1, y0, y1 = box
    t_in = [t for t in truth if x0 <= t[1] <= x1 and y0 <= t[2] <= y1]
    pairs = sorted(
        (math.hypot(b[1] - t[1], b[2] - t[2]), bi, ti)
        for bi, b in enumerate(baseline) for ti, t in enumerate(t_in)
        if math.hypot(b[1] - t[1], b[2] - t[2]) <= radius)
    ub, ut = set(), set()
    for _, bi, ti in pairs:
        if bi not in ub and ti not in ut:
            ub.add(bi)
            ut.add(ti)
    return {'truth_in_box': len(t_in), 'baseline': len(baseline),
            'tp': len(ub), 'fp': len(baseline) - len(ub),
            'fn': len(t_in) - len(ut)}


def format_report(result: Dict, meta: Dict) -> str:
    lines = [f"# Removal test — {'PASS' if result['passed'] else 'FAIL'}", '']
    for k, v in meta.items():
        lines.append(f'- {k}: {v}')
    lines += ['',
              f"**Right:** {result['true_positives']}/{result['removed']} "
              f"removed trees reported LOST.  **Missed:** {result['missed']}."
              f"  **Wrong:** {result['false_events']} false events."]
    if result.get('understory_trees') or result.get('crown_shared_trees'):
        lines += ['', f"**Tiers:** canopy {result['canopy_found']}/"
                  f"{result['canopy_trees']} LOST (tree level) · crown-shared "
                  f"{result.get('crown_shared_ok', 0)}/"
                  f"{result.get('crown_shared_trees', 0)} LOST or area alert · "
                  f"understory {result['understory_ok']}/"
                  f"{result['understory_trees']} area alert or not observable."]
    lines += ['', '| tree | tier | canopy over trunk before→after | in baseline | result | detail |',
              '|---|---|---|---|---|---|']
    for t in result['trees']:
        base = (f"yes (#{t['baseline_id']}, {t['baseline_dist']:.2f} m)"
                if t['in_baseline'] else
                f"no ({t['baseline_dist']:.2f} m)"
                if t['baseline_dist'] is not None else 'no')
        if t['detected']:
            res = 'LOST ✓'
            det = (f"#{t['event_id']} after {t['latency_s']:.0f} s, "
                   f"{t['position_error']:.2f} m off")
        elif t.get('tier') in ('understory', 'crown-shared') and t.get('area_alert'):
            res = 'AREA ALERT ✓'
            det = t['tier'] + ': canopy-loss alert within 3 m'
        elif t.get('tier') == 'understory' and t.get('tier_ok'):
            res = 'NOT OBSERVABLE'
            det = ('understory: canopy over the trunk did not drop beyond '
                   'measurement noise — no top-down sensor can see it')
        elif t.get('tier') == 'unobserved':
            res = 'NOT SCORED'
            det = 'never scanned before the removal'
        else:
            res = 'MISSED ✗'
            det = t['why_missed']
        tb, ta = t.get('top_before'), t.get('top_after')
        tops = (f"{tb:.1f} → {ta:.1f} m" if tb is not None and ta is not None
                else '–')
        lines.append(f"| {t['name']} ({t['x']:.1f}, {t['y']:.1f}) | "
                     f"{t.get('tier', 'canopy')} | {tops} | {base} | {res} | {det} |")
    if result['false_event_list']:
        lines += ['', '**False events:**', '']
        for e in result['false_event_list']:
            if e['type'] == 'AREA':
                lines.append(
                    f"- AREA alert at ({e['x']:.1f}, {e['y']:.1f}) — nearest "
                    f"removed tree {e['nearest_tree']} {e['nearest_dist']:.1f} "
                    f"m: {e['why']}")
                continue
            lines.append(
                f"- {e['type']} #{e['id']} at ({e['x']:.1f}, {e['y']:.1f}) "
                f"after {e['t']:.0f} s — nearest tree {e['nearest_tree']} "
                f"{e['nearest_dist']:.2f} m: {e['why']}")
    lines += ['', f"scan_mapper after removal: {result['canopy_lost_events']} "
              f"CANOPY LOST, {result['canopy_new_events']} NEW CANOPY events."]
    return '\n'.join(lines) + '\n'


# ── ROS node ────────────────────────────────────────────────────────

def _cloud_xyz(cloud) -> np.ndarray:
    offs = {f.name: f.offset for f in cloud.fields}
    if not all(k in offs for k in 'xyz') or cloud.width == 0:
        return np.zeros((0, 3), dtype=np.float32)
    raw = np.frombuffer(bytes(cloud.data), dtype=np.uint8)
    arr = raw[:cloud.width * cloud.point_step].reshape(-1, cloud.point_step)
    cols = [arr[:, offs[k]:offs[k] + 4].copy().view('<f4').ravel()
            for k in 'xyz']
    return np.stack(cols, axis=1)


def main(args=None):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import (QoSProfile, ReliabilityPolicy, DurabilityPolicy,
                           HistoryPolicy)
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import String
    from deforestation_interfaces.msg import SuspiciousArea, TreeChangeEvent
    from nav_msgs.msg import OccupancyGrid
    from rcl_interfaces.msg import ParameterDescriptor, ParameterType
    from ament_index_python.packages import get_package_share_directory

    from .tree_detection import parse_tree_truth
    from .simulate_tree_removal import remove_model
    from .visibility import classify_tree, disc_max, removal_tier

    class RemovalTest(Node):
        def __init__(self):
            super().__init__('removal_test')
            p = self.declare_parameter
            p('world', 'dense_forest')
            p('n_trees', 10)
            p('balanced', False)
            p('canopy_only', False)
            p('tree_names', [''], ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING_ARRAY))
            p('trees', '')      # the same as a comma-separated string (launch argument)
            for k, v in (('survey_x_min', -30.0), ('survey_x_max', 30.0),
                         ('survey_y_min', -30.0), ('survey_y_max', 30.0)):
                p(k, v)
            p('margin', 5.0)
            p('min_separation', 10.0)
            p('min_coverage', 0.9)
            p('match_radius', 1.5)
            p('settle_loops', 2)
            p('max_wait_s', 2400.0)
            p('report_dir', '/tmp/deforestation_eval')
            g = lambda k: self.get_parameter(k).value  # noqa: E731

            self.world = g('world')
            self.box = (g('survey_x_min'), g('survey_x_max'),
                        g('survey_y_min'), g('survey_y_max'))
            sdf = os.path.join(get_package_share_directory(
                '41068_ignition_bringup'), 'worlds', f'{self.world}.sdf')
            self.truth = parse_tree_truth(sdf)
            names = [n for n in g('tree_names') if n] + \
                [n.strip() for n in g('trees').split(',') if n.strip()]
            if names:
                by_name = {t[0]: t for t in self.truth}
                self.targets = [by_name[n] for n in names if n in by_name]
            else:
                self.targets = select_removal_targets(
                    self.truth, int(g('n_trees')), self.box,
                    g('margin'), g('min_separation'),
                    balanced=bool(g('balanced')),
                    canopy_only=bool(g('canopy_only')))
            self.min_cov = float(g('min_coverage'))
            self.match_radius = float(g('match_radius'))
            self.settle_loops = int(g('settle_loops'))
            self.max_wait = float(g('max_wait_s'))
            self.report_dir = g('report_dir')

            self.coverage = 0.0
            self.baseline = None            # [(id, x, y)]
            self.detections: List[Point2] = []
            self.events: List[Dict] = []
            self.canopy: List[Tuple[str, float, float]] = []
            self.removed: List[Tree] = []
            self.map = None          # latest (grid [ix,iy] m, res, origin)
            self.pre_map = None      # canopy map at the moment of removal
            self.flags: List[Point2] = []
            self.removal_failed: List[str] = []
            self.t_removed = None
            self.t_start = time.time()
            self.loop_counter = LoopCounter()
            self.state = 'WAIT_BASELINE'
            self.done = False

            latched = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST, depth=1)
            rel = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST, depth=50)
            self.create_subscription(PointCloud2, '/parrot_tree_baseline',
                                     self._baseline_cb, latched)
            self.create_subscription(PointCloud2, '/parrot_tree_positions',
                                     self._positions_cb, rel)
            self.create_subscription(String, '/scan_coverage',
                                     self._coverage_cb, rel)
            self.create_subscription(TreeChangeEvent,
                                     '/parrot_tree_change_events',
                                     self._event_cb, rel)
            self.create_subscription(String, '/canopy_change_events',
                                     self._canopy_cb, rel)
            self.create_subscription(OccupancyGrid, '/forest_canopy_map',
                                     self._map_cb, latched)
            self.create_subscription(SuspiciousArea, '/suspicious_areas',
                                     self._flag_cb, rel)
            self.create_subscription(String, '/survey_status',
                                     self._survey_cb, rel)
            self.result_pub = self.create_publisher(
                String, '/removal_test_result', latched)
            self.create_timer(1.0, self._tick)
            self.get_logger().info(
                f'Removal test ready: world={self.world}, '
                f'{len(self.targets)} targets: '
                + ', '.join(f'{n}({x:.1f},{y:.1f})'
                            for n, x, y in self.targets))

        # -- callbacks --
        def _baseline_cb(self, msg):
            pts = _cloud_xyz(msg)
            self.baseline = [(int(round(z)), float(x), float(y))
                             for x, y, z in pts]

        def _positions_cb(self, msg):
            self.detections = [(float(x), float(y))
                               for x, y, _ in _cloud_xyz(msg)]

        def _coverage_cb(self, msg):
            m = re.search(r'coverage=([0-9.]+)%', msg.data)
            if m:
                self.coverage = float(m.group(1)) / 100.0

        def _since(self):
            return 0.0 if self.t_removed is None else time.time() - self.t_removed

        def _event_cb(self, msg):
            e = {'type': msg.event_type, 'id': int(msg.tree_id),
                 'x': float(msg.x), 'y': float(msg.y), 't': self._since(),
                 'before_removal': self.t_removed is None}
            self.events.append(e)
            self.get_logger().info(
                f'[test] event {e["type"]} #{e["id"]} at '
                f'({e["x"]:.1f},{e["y"]:.1f}) t={e["t"]:.0f}s')

        def _map_cb(self, msg):
            g = np.array(msg.data, dtype=np.float32).reshape(
                msg.info.height, msg.info.width).T
            self.map = (np.where(g >= 0, g / 10.0, -1.0), msg.info.resolution,
                        (msg.info.origin.position.x, msg.info.origin.position.y))

        def _flag_cb(self, msg):
            if self.t_removed is not None:
                self.flags.append((float(msg.position.x), float(msg.position.y)))

        def _canopy_cb(self, msg):
            if self.t_removed is not None:
                self.canopy.extend(parse_canopy_events(msg.data))

        @property
        def loops_after(self):
            return self.loop_counter.loops

        def _survey_cb(self, msg):
            if self.t_removed is None:
                return
            if self.loop_counter.update_from_status(msg.data):
                self.get_logger().info(
                    f'[test] survey loop {self.loops_after}/'
                    f'{self.settle_loops} completed after removal')

        # -- state machine --
        def _tick(self):
            if self.done:
                return
            if time.time() - self.t_start > self.max_wait:
                self.get_logger().error('[test] max_wait_s exceeded')
                self._finish(timeout=True)
                return
            if self.state == 'WAIT_BASELINE':
                if self.baseline is not None and self.coverage >= self.min_cov:
                    self._remove()
            elif self.state == 'MONITOR':
                if self.loops_after >= self.settle_loops:
                    self._finish()

        def _remove(self):
            self.get_logger().info(
                f'[test] baseline frozen ({len(self.baseline)} trees), '
                f'coverage {self.coverage * 100:.1f}% — removing '
                f'{len(self.targets)} trees')
            for name, x, y in self.targets:
                ok, out = remove_model(self.world, name)
                if ok:
                    self.removed.append((name, x, y))
                else:
                    self.removal_failed.append(name)
                    self.get_logger().error(f'[test] remove {name}: {out}')
            try:
                listing = subprocess.run(
                    ['ign', 'model', '--list'], capture_output=True,
                    text=True, timeout=20.0).stdout
                still = [n for n, _, _ in self.removed
                         if re.search(rf'-\s+{re.escape(n)}\s*$', listing, re.M)]
                if still:
                    self.get_logger().error(f'[test] still in world: {still}')
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f'[test] model list failed: {exc}')
            self.pre_map = self.map
            self.t_removed = time.time()
            self.state = 'MONITOR'
            self.get_logger().info(
                f'[test] removed {len(self.removed)}: '
                + ', '.join(n for n, _, _ in self.removed))

        def _finish(self, timeout=False):
            self.done = True
            after = [e for e in self.events if not e['before_removal']]
            before = [e for e in self.events if e['before_removal']]
            tiers = {}
            if self.pre_map is not None:
                g0, res0, org0 = self.pre_map
                for tree in self.removed:
                    cls, top0 = classify_tree(g0, res0, org0, tree)
                    cls = removal_tier(tree, cls, self.truth)
                    top1 = (disc_max(self.map[0], self.map[1], self.map[2],
                                     tree[1], tree[2], 0.75)
                            if self.map is not None else None)
                    tiers[tree[0]] = {'class': cls, 'top_before': top0,
                                      'top_after': top1}
            area = [(c[1], c[2]) for c in self.canopy if c[0] == 'LOST'] + self.flags
            result = score_removal(
                self.removed, self.truth, self.baseline or [], after,
                self.canopy, self.detections, self.match_radius,
                tiers=tiers or None, area_alerts=area)
            for e in before:   # events before removal are false by definition
                result['false_event_list'].append(
                    {**e, 'nearest_tree': None, 'nearest_dist': float('nan'),
                     'nearest_was_removed': False,
                     'why': 'event before any tree was removed'})
            result['false_events'] = len(result['false_event_list'])
            result['passed'] = result['passed'] and not before and not timeout
            quality = baseline_quality(self.truth, self.baseline or [],
                                       self.box)
            meta = {
                'world': self.world,
                'coverage at removal': f'{self.coverage * 100:.1f}%',
                'baseline vs SDF (1 m)': (
                    f"{quality['baseline']} trees, TP {quality['tp']}, "
                    f"FP {quality['fp']}, FN {quality['fn']} of "
                    f"{quality['truth_in_box']} in the survey box"),
                'removal failed': self.removal_failed or 'none',
                'observed after removal': (
                    f'{self._since():.0f} s, {self.loops_after} survey loops'
                    + (' (TIMEOUT)' if timeout else '')),
            }
            report = format_report(result, meta)
            os.makedirs(self.report_dir, exist_ok=True)
            stamp = time.strftime('%Y%m%d_%H%M%S')
            base = os.path.join(self.report_dir, f'removal_test_{stamp}')
            with open(base + '.md', 'w') as fh:
                fh.write(report)
            with open(base + '.json', 'w') as fh:
                json.dump({'meta': meta, 'baseline_quality': quality,
                           'world': self.world,
                           't_removed_epoch': self.t_removed,
                           'removed': self.removed,
                           'result': result, 'baseline': self.baseline,
                           'events': self.events}, fh, indent=2, default=str)
            verdict = (f"{'PASS' if result['passed'] else 'FAIL'}: "
                       f"{result['true_positives']}/{result['removed']} "
                       f"LOST, {result['missed']} missed, "
                       f"{result['false_events']} false — {base}.md")
            self.result_pub.publish(String(data=verdict))
            self.get_logger().info('[test] ' + verdict)
            for line in report.splitlines():
                self.get_logger().info('[report] ' + line)

    rclpy.init(args=args)
    node = RemovalTest()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.5)
        # let the latched verdict go out
        end = time.time() + 2.0
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
