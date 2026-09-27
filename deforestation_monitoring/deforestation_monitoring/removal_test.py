#!/usr/bin/env python3
"""Live tree removal acceptance test.

Run against a live simulation (see ``removal_test.launch.py``):

  1. wait until the drone survey reaches ``min_coverage`` (default 90%) and
     ``parrot_tree_tracker`` has frozen its tree baseline;
  2. delete ``n_trees`` trees from the Gazebo world (default 10, picked
     automatically inside the survey box and well apart);
  3. keep patrolling for ``settle_loops`` survey loops;
  4. score what the pipeline reported against the known removals:

     * right: a LOST event at a removed tree (true positive);
     * missed: a removed tree with no LOST event (false negative), with the
       likely reason;
     * wrong: a LOST event at a standing tree, or any GAINED event (false
       positive).

A markdown and a JSON report are written to ``report_dir`` and a one-line
verdict is published on ``/removal_test_result``. PASS means every removed
tree was reported LOST and there were no false events.

The scoring functions have no ROS dependency, so they are unit-tested
offline.
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

Tree = Tuple[str, float, float]            # (name, x, y)
Point2 = Tuple[float, float]


# Pure logic

def occluded_from_above(truth: Sequence[Tree],
                        radius: float = 7.0) -> set:
    """Pines within ``radius`` of an oak trunk (possibly under its crown).

    Optional filter, not used by default.
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
) -> List[Tree]:
    """Pick ``n`` trees inside ``box`` (x_min, x_max, y_min, y_max) shrunk
    by ``margin``: most isolated first and at least ``min_separation`` apart,
    so each removal is scored on its own. With ``visible_only``, trees that
    may be under another crown are skipped. The result is deterministic."""
    x0, x1, y0, y1 = box
    hidden = occluded_from_above(truth) if visible_only else set()
    inside = [t for t in truth
              if x0 + margin <= t[1] <= x1 - margin
              and y0 + margin <= t[2] <= y1 - margin
              and t[0] not in hidden]

    def nn(t):
        return min((math.hypot(t[1] - o[1], t[2] - o[2])
                    for o in truth if o[0] != t[0]), default=float('inf'))

    ranked = sorted(inside, key=lambda t: (-nn(t), t[0]))
    if balanced:
        # Alternate species (oak, pine, oak, ...) so a demo removes both;
        # within a species the most isolated come first.
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


class LoopCounter:
    """Counts full survey loops from the ``/survey_status`` waypoint index.

    When the waypoint index goes down (a wrap), it only counts as a loop if
    at least half of the waypoints were visited since the last wrap. A wrap
    right after the removal therefore does not count as a whole loop.
    """

    def __init__(self):
        self.loops = 0
        self._last = None
        self._visited = set()

    def update(self, wp: int, total: int) -> bool:
        wrapped = self._last is not None and wp < self._last
        counted = False
        if wrapped:
            counted = len(self._visited) >= total / 2
            self.loops += int(counted)
            self._visited = set()
        self._visited.add(wp)
        self._last = wp
        return counted


def parse_canopy_events(text: str) -> List[Tuple[str, float, float]]:
    """Parse ``/canopy_change_events`` strings into [(LOST|NEW, x, y)]."""
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
) -> Dict:
    """Score tracker events against the known removals.

    ``baseline``: [(id, x, y)] frozen baseline trees.
    ``events``: [{'type': 'LOST'|'GAINED', 'id', 'x', 'y', 't'}], where t is
    seconds after the removal.
    """
    base_pts = [(b[1], b[2]) for b in baseline]
    lost = [e for e in events if e['type'] == 'LOST']
    gained = [e for e in events if e['type'] == 'GAINED']

    # One-to-one: each removed tree takes its closest unclaimed LOST event.
    pairs = []
    for ri, (_, rx, ry) in enumerate(removed):
        for ei, e in enumerate(lost):
            d = math.hypot(e['x'] - rx, e['y'] - ry)
            if d <= match_radius:
                pairs.append((d, ri, ei))
    pairs.sort()
    tree_event: Dict[int, int] = {}
    used = set()
    for d, ri, ei in pairs:
        if ri in tree_event or ei in used:
            continue
        tree_event[ri] = ei
        used.add(ei)

    trees = []
    for ri, (name, rx, ry) in enumerate(removed):
        bi, bd = _nearest((rx, ry), base_pts)
        in_baseline = bi is not None and bd <= baseline_radius
        entry = {
            'name': name, 'x': rx, 'y': ry,
            'in_baseline': in_baseline,
            'baseline_id': baseline[bi][0] if in_baseline else None,
            'baseline_dist': bd if bi is not None else None,
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

    removed_names = {r[0] for r in removed}
    false_events = []
    for ei, e in enumerate(lost):
        if ei in used:
            continue
        ti, td = _nearest((e['x'], e['y']), [(t[1], t[2]) for t in truth])
        near = truth[ti][0] if ti is not None else None
        false_events.append({
            **e, 'nearest_tree': near, 'nearest_dist': td,
            'nearest_was_removed': near in removed_names,
            'why': ('duplicate / off-position LOST for a removed tree'
                    if near in removed_names and td <= 3.0 else
                    'LOST for a tree that is still standing'),
        })
    for e in gained:
        ti, td = _nearest((e['x'], e['y']), [(t[1], t[2]) for t in truth])
        false_events.append({
            **e, 'nearest_tree': truth[ti][0] if ti is not None else None,
            'nearest_dist': td, 'nearest_was_removed': False,
            'why': 'GAINED — nothing was added to the world',
        })

    tp = sum(1 for t in trees if t['detected'])
    fn = len(trees) - tp
    fp = len(false_events)
    return {
        'removed': len(removed),
        'true_positives': tp,
        'missed': fn,
        'false_events': fp,
        'passed': fn == 0 and fp == 0 and len(removed) > 0,
        'trees': trees,
        'false_event_list': false_events,
        'canopy_new_events': sum(1 for c in canopy_events if c[0] == 'NEW'),
        'canopy_lost_events': sum(1 for c in canopy_events if c[0] == 'LOST'),
    }


def baseline_quality(truth: Sequence[Tree],
                     baseline: Sequence[Tuple[int, float, float]],
                     box: Tuple[float, float, float, float],
                     radius: float = 1.0) -> Dict:
    """Baseline vs SDF inside the survey box (greedy one-to-one match)."""
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
              f"  **Wrong:** {result['false_events']} false events.",
              '', '| tree | position | in baseline | result | detail |',
              '|---|---|---|---|---|']
    for t in result['trees']:
        base = (f"yes (#{t['baseline_id']}, {t['baseline_dist']:.2f} m)"
                if t['in_baseline'] else
                f"no ({t['baseline_dist']:.2f} m)"
                if t['baseline_dist'] is not None else 'no')
        if t['detected']:
            res = 'LOST ✓'
            det = (f"#{t['event_id']} after {t['latency_s']:.0f} s, "
                   f"{t['position_error']:.2f} m off")
        else:
            res = 'MISSED ✗'
            det = t['why_missed']
        lines.append(f"| {t['name']} | ({t['x']:.1f}, {t['y']:.1f}) | "
                     f"{base} | {res} | {det} |")
    if result['false_event_list']:
        lines += ['', '**False events:**', '']
        for e in result['false_event_list']:
            lines.append(
                f"- {e['type']} #{e['id']} at ({e['x']:.1f}, {e['y']:.1f}) "
                f"after {e['t']:.0f} s — nearest tree {e['nearest_tree']} "
                f"{e['nearest_dist']:.2f} m: {e['why']}")
    lines += ['', f"scan_mapper after removal: {result['canopy_lost_events']} "
              f"CANOPY LOST, {result['canopy_new_events']} NEW CANOPY events."]
    return '\n'.join(lines) + '\n'


# ROS node

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
    from deforestation_interfaces.msg import TreeChangeEvent
    from rcl_interfaces.msg import ParameterDescriptor, ParameterType
    from ament_index_python.packages import get_package_share_directory

    from .tree_detection import parse_tree_truth
    from .simulate_tree_removal import remove_model

    class RemovalTest(Node):
        def __init__(self):
            super().__init__('removal_test')
            p = self.declare_parameter
            p('world', 'dense_forest')
            p('n_trees', 10)
            p('balanced', False)
            p('tree_names', [''], ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING_ARRAY))
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
            names = [n for n in g('tree_names') if n]
            if names:
                by_name = {t[0]: t for t in self.truth}
                self.targets = [by_name[n] for n in names if n in by_name]
            else:
                self.targets = select_removal_targets(
                    self.truth, int(g('n_trees')), self.box,
                    g('margin'), g('min_separation'),
                    balanced=bool(g('balanced')))
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

        # Callbacks
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

        def _canopy_cb(self, msg):
            if self.t_removed is not None:
                self.canopy.extend(parse_canopy_events(msg.data))

        @property
        def loops_after(self):
            return self.loop_counter.loops

        def _survey_cb(self, msg):
            m = re.search(r'STATE=PATROL wp=(\d+)/(\d+)', msg.data)
            if not m or self.t_removed is None:
                return
            if self.loop_counter.update(int(m.group(1)), int(m.group(2))):
                self.get_logger().info(
                    f'[test] survey loop {self.loops_after}/'
                    f'{self.settle_loops} completed after removal')

        # State machine
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
            self.t_removed = time.time()
            self.state = 'MONITOR'
            self.get_logger().info(
                f'[test] removed {len(self.removed)}: '
                + ', '.join(n for n, _, _ in self.removed))

        def _finish(self, timeout=False):
            self.done = True
            after = [e for e in self.events if not e['before_removal']]
            before = [e for e in self.events if e['before_removal']]
            result = score_removal(
                self.removed, self.truth, self.baseline or [], after,
                self.canopy, self.detections, self.match_radius)
            for e in before:   # any event before the removal is false
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
        # Give the latched verdict time to go out
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
