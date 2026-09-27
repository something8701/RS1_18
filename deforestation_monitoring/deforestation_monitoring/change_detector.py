#!/usr/bin/env python3
"""
Canopy Change Detector: compares the current drone canopy height map against
a frozen baseline and reports areas of significant height loss.

Pipeline position:
    scan_mapper  --/forest_canopy_map-->  change_detector  --/canopy_change_map-->
    pattern_scanner --/suspicious_areas--> mission_coordinator (Husky, Nav2)

Input:
  /forest_canopy_map  OccupancyGrid, value = height * height_scale (0-100),
                      -1 = never scanned. Published by scan_mapper.

Outputs (same topics/formats scan_mapper used, so downstream nodes and the
dashboard are unchanged):
  /canopy_change_map      OccupancyGrid. Lost cells = -(percent of baseline
                          height lost), 1..100. Gained cells = +percent.
                          0 = unchanged or not comparable.
  /canopy_change_events   String alerts, one per newly detected cluster
  /canopy_change_markers  MarkerArray (red = lost, blue = gained)
  /drone_baseline_status  String 'baseline_established:<canopy cells>'
                          (simulate_tree_removal waits on this)
  ~/status                String, periodic phase / coverage / change summary

Baseline:
  During the BASELINE phase the detector keeps the per-cell maximum height
  seen. It freezes that as the baseline when the trigger fires:
    survey   (default) first lawnmower pass complete, from /survey_status
    coverage fraction of the survey region scanned >= coverage_fraction
    manual   only via the ~/snapshot_baseline service
  The baseline can be saved to / loaded from an .npz file, so a later run
  can be compared against an earlier one.

A cell counts as LOST when all of these hold:
  1. it was canopy in the baseline         (baseline >= canopy_threshold)
  2. its height fell by >= height_drop_threshold metres
  3. it is now below canopy height         (if require_below_canopy)
  4. it has met 1-3 for >= persistence_updates consecutive map updates
Scan-line gaps: the drone lidar is a 2 Hz push-broom, so at 3 m/s its scan
lines are ~1.5 m apart and the canopy map is striped (one 0.25 m cell wide
line every ~6 cells). Lost cells therefore also arrive as parallel stripes.
Before clustering, stripes within gap_bridge_m of each other are joined with
a morphological closing, so a removed crown becomes one region instead of
several 1-cell-wide lines. A region is reported only if it contains at least
min_cluster_cells *measured* lost cells (bridged fill cells don't count), so
a couple of isolated misreads can never add up to a detection.

The detection maths lives in ChangeCore (numpy only, no ROS) so it can be
unit tested without a simulator.
"""

import math
import os
import re

import numpy as np
from scipy import ndimage


# ─────────────────────────────────────────────────────────────────────────
# ROS-free core
# ─────────────────────────────────────────────────────────────────────────

class ChangeCore:
    """Baseline storage and change detection on a 2D height grid.

    Grids are (rows, cols) = (iy, ix), matching OccupancyGrid row-major
    order, with heights in metres and NaN for never-scanned cells.
    """

    def __init__(self, canopy_threshold=2.0, height_drop=3.0,
                 require_below_canopy=True, persistence_updates=3,
                 gap_bridge_cells=8, min_cluster_cells=8):
        self.canopy_threshold = float(canopy_threshold)
        self.height_drop = float(height_drop)
        self.require_below_canopy = bool(require_below_canopy)
        self.persistence = max(1, int(persistence_updates))
        self.gap_bridge_cells = max(0, int(gap_bridge_cells))
        self.min_cluster_cells = max(1, int(min_cluster_cells))

        self.accum = None       # running max during BASELINE phase
        self.baseline = None    # frozen baseline heights
        self.lost_streak = None
        self.gain_streak = None

    # -- baseline --------------------------------------------------------

    @property
    def has_baseline(self):
        return self.baseline is not None

    def reset(self):
        self.accum = None
        self.baseline = None
        self.lost_streak = None
        self.gain_streak = None

    def accumulate(self, heights):
        """Fold a new height grid into the baseline candidate (per-cell max)."""
        if self.accum is None or self.accum.shape != heights.shape:
            self.accum = heights.astype(np.float32).copy()
        else:
            # fmax ignores NaN, so unscanned cells never erase scanned ones.
            self.accum = np.fmax(self.accum, heights)

    def freeze(self):
        """Freeze the accumulated candidate as the baseline."""
        if self.accum is None:
            return False
        self.set_baseline(self.accum)
        return True

    def set_baseline(self, heights):
        self.baseline = heights.astype(np.float32).copy()
        self.lost_streak = np.zeros(heights.shape, dtype=np.int16)
        self.gain_streak = np.zeros(heights.shape, dtype=np.int16)

    def baseline_canopy_cells(self):
        if self.baseline is None:
            return 0
        with np.errstate(invalid='ignore'):
            return int(np.count_nonzero(self.baseline >= self.canopy_threshold))

    # -- change detection --------------------------------------------------

    def update(self, current):
        """Compare the current grid to the baseline.

        Returns (lost_mask, gained_mask, loss_pct, gain_pct): the measured
        cells that passed the persistence check, and per-cell percentages
        (0..100) of height lost / gained, valid where the mask is True.
        Pass each mask to regions() to bridge scan-line gaps and gate by size.
        """
        b = self.baseline
        with np.errstate(invalid='ignore'):
            comparable = np.isfinite(b) & np.isfinite(current)
            drop = b - current
            b_canopy = b >= self.canopy_threshold
            c_canopy = current >= self.canopy_threshold

            lost_raw = comparable & b_canopy & (drop >= self.height_drop)
            if self.require_below_canopy:
                lost_raw &= ~c_canopy
            gained_raw = (comparable & ~b_canopy & c_canopy
                          & (-drop >= self.height_drop))

        # Persistence: a cell must hold its state across consecutive updates.
        self.lost_streak = np.where(lost_raw, self.lost_streak + 1, 0).astype(np.int16)
        self.gain_streak = np.where(gained_raw, self.gain_streak + 1, 0).astype(np.int16)
        lost = self.lost_streak >= self.persistence
        gained = self.gain_streak >= self.persistence

        with np.errstate(invalid='ignore', divide='ignore'):
            loss_pct = np.where(lost, np.clip(100.0 * drop / b, 0.0, 100.0), 0.0)
            gain_pct = np.where(
                gained,
                np.clip(100.0 * (-drop) / np.maximum(current, 1e-3), 0.0, 100.0),
                0.0)
        return lost, gained, loss_pct, gain_pct

    def regions(self, mask, pct, change_m=None):
        """Bridge scan-line gaps, label regions and gate them by measured size.

        Returns (region_mask, region_pct, clusters):
          region_mask  cells of every accepted region, gaps filled
          region_pct   per-cell percent: measured cells keep their own value,
                       filled gap cells get the region's mean
          clusters     one dict per accepted region, largest first, with
                       row/col centroid, measured cell count, region cell
                       count, mean percent and (optional) mean change in m
        """
        region_mask = np.zeros(mask.shape, dtype=bool)
        region_pct = np.zeros(mask.shape, dtype=np.float32)
        if not np.any(mask):
            return region_mask, region_pct, []

        bridged = mask
        if self.gap_bridge_cells > 0:
            k = 2 * self.gap_bridge_cells + 1
            # Pad so closing doesn't erode regions touching the grid edge.
            pad = self.gap_bridge_cells
            padded = np.pad(mask, pad)
            # A k x k square closing done as separate row and column passes:
            # identical result, ~5x less CPU than the 2D structuring element.
            col = np.ones((k, 1), dtype=bool)
            row = np.ones((1, k), dtype=bool)
            dil = ndimage.binary_dilation(ndimage.binary_dilation(padded, col), row)
            closed = ndimage.binary_erosion(ndimage.binary_erosion(dil, col), row)
            bridged = closed[pad:-pad, pad:-pad] | mask

        labels, n = ndimage.label(bridged, structure=np.ones((3, 3), dtype=bool))
        clusters = []
        for lid in range(1, n + 1):
            region = labels == lid
            measured = region & mask
            n_measured = int(np.count_nonzero(measured))
            if n_measured < self.min_cluster_cells:
                continue
            mean_pct = float(np.mean(pct[measured]))
            region_mask |= region
            region_pct[region] = mean_pct
            region_pct[measured] = pct[measured]
            r, c = ndimage.center_of_mass(region)
            entry = {'row': float(r), 'col': float(c),
                     'cells': n_measured,
                     'region_cells': int(np.count_nonzero(region)),
                     'mean_pct': mean_pct}
            if change_m is not None:
                entry['mean_change_m'] = float(np.nanmean(change_m[measured]))
            clusters.append(entry)
        clusters.sort(key=lambda d: d['region_cells'], reverse=True)
        return region_mask, region_pct, clusters


def parse_survey_status(text):
    """Parse demo_patrol's /survey_status. Returns (wp, total, coverage) or None."""
    if 'STATE=PATROL' not in text:
        return None
    m_wp = re.search(r'wp=(\d+)/(\d+)', text)
    m_cov = re.search(r'coverage=(\d+(?:\.\d+)?)%', text)
    if not (m_wp and m_cov):
        return None
    return int(m_wp.group(1)), int(m_wp.group(2)), float(m_cov.group(1))


# ─────────────────────────────────────────────────────────────────────────
# ROS node
# ─────────────────────────────────────────────────────────────────────────

def main(args=None):
    import rclpy
    rclpy.init(args=args)
    node = _make_node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _make_node():
    # ROS imports are kept inside so ChangeCore can be imported and unit
    # tested on a machine without ROS installed.
    from rclpy.node import Node
    from rcl_interfaces.msg import ParameterDescriptor
    from rclpy.qos import (QoSProfile, ReliabilityPolicy, DurabilityPolicy,
                           HistoryPolicy)
    from nav_msgs.msg import OccupancyGrid
    from geometry_msgs.msg import Point, Pose, Quaternion
    from std_msgs.msg import Header, String
    from std_srvs.srv import Trigger
    from visualization_msgs.msg import Marker, MarkerArray

    def desc(text):
        return ParameterDescriptor(description=text)

    class ChangeDetector(Node):

        def __init__(self):
            super().__init__('change_detector')

            p = self.declare_parameter
            p('canopy_topic', '/forest_canopy_map', desc('Canopy height map input'))
            p('height_scale', 10.0, desc('Map value per metre (scan_mapper height_scale)'))
            p('canopy_threshold', 2.0, desc('Height (m) at or above which a cell is canopy'))
            p('height_drop_threshold', 1.5, desc('Minimum height loss (m) for a lost cell'))
            p('require_below_canopy', True,
              desc('Lost cells must also now read below canopy_threshold'))
            p('persistence_updates', 3,
              desc('Consecutive map updates a cell must stay lost/gained'))
            p('gap_bridge_m', 2.0,
              desc('Join lost cells up to this far apart (m) into one region. '
                   'Must exceed the lidar scan-line spacing (speed / scan rate).'))
            p('min_cluster_cells', 8, desc('Minimum connected cells to report a cluster'))
            p('publish_gained', True, desc('Also report new canopy (regrowth)'))
            p('alert_bucket_m', 3.0,
              desc('Clusters within the same bucket (m) are alerted only once'))
            p('baseline_trigger', 'survey', desc('survey | coverage | manual'))
            p('survey_coverage_pct', 100.0,
              desc('survey trigger: /survey_status coverage (%) that ends the first pass'))
            p('region_x_min', -15.0, desc('Survey region for coverage stats (m)'))
            p('region_x_max', 15.0, desc('Survey region for coverage stats (m)'))
            p('region_y_min', -15.0, desc('Survey region for coverage stats (m)'))
            p('region_y_max', 15.0, desc('Survey region for coverage stats (m)'))
            p('coverage_fraction', 0.9,
              desc('coverage trigger: fraction of region cells scanned'))
            p('min_baseline_coverage', 0.3,
              desc('Refuse an automatic baseline if less than this fraction of the '
                   'survey region has been scanned (e.g. the drone lidar was not '
                   'publishing); wait for the next pass instead'))
            p('baseline_file', '/tmp/deforestation_eval/canopy_baseline.npz',
              desc('Where the baseline is saved / loaded'))
            p('save_baseline', True, desc('Save the baseline to baseline_file when frozen'))
            p('load_baseline', False,
              desc('Load baseline_file at startup (skips the BASELINE phase)'))
            p('status_period', 5.0, desc('Seconds between ~/status publishes'))

            g = lambda n: self.get_parameter(n).value  # noqa: E731
            self.height_scale = float(g('height_scale'))
            self.publish_gained = bool(g('publish_gained'))
            self.bucket_m = float(g('alert_bucket_m'))
            self.trigger = str(g('baseline_trigger'))
            if self.trigger not in ('survey', 'coverage', 'manual'):
                self.get_logger().warn(f'Unknown baseline_trigger "{self.trigger}", using survey')
                self.trigger = 'survey'
            self.survey_pct = float(g('survey_coverage_pct'))
            self.region = (g('region_x_min'), g('region_x_max'),
                           g('region_y_min'), g('region_y_max'))
            self.coverage_fraction = float(g('coverage_fraction'))
            self.min_baseline_coverage = float(g('min_baseline_coverage'))
            self.baseline_file = str(g('baseline_file'))
            self.save_baseline = bool(g('save_baseline'))
            self.load_baseline = bool(g('load_baseline'))

            self.gap_bridge_m = float(g('gap_bridge_m'))
            self.core = ChangeCore(
                canopy_threshold=g('canopy_threshold'),
                height_drop=g('height_drop_threshold'),
                require_below_canopy=g('require_below_canopy'),
                persistence_updates=g('persistence_updates'),
                gap_bridge_cells=0,   # set from gap_bridge_m once resolution is known
                min_cluster_cells=g('min_cluster_cells'),
            )

            self.geom = None           # (w, h, res, ox, oy)
            self.frame_id = ''
            self.last_info = None
            self.coverage = 0.0
            self.updates = 0
            self._pending_load = self.load_baseline
            self._last_wp = None
            self._alerted_lost = set()
            self._alerted_gained = set()
            self._n_lost_clusters = 0
            self._n_gained_clusters = 0
            self._lost_cells = 0

            map_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 history=HistoryPolicy.KEEP_LAST, depth=1)
            default_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                     durability=DurabilityPolicy.VOLATILE,
                                     history=HistoryPolicy.KEEP_LAST, depth=10)

            self.create_subscription(OccupancyGrid, g('canopy_topic'),
                                     self.map_cb, map_qos)
            self.create_subscription(String, '/survey_status',
                                     self.survey_cb, default_qos)

            self.change_pub = self.create_publisher(
                OccupancyGrid, '/canopy_change_map', map_qos)
            self.events_pub = self.create_publisher(
                String, '/canopy_change_events', default_qos)
            self.marker_pub = self.create_publisher(
                MarkerArray, '/canopy_change_markers', default_qos)
            self.baseline_pub = self.create_publisher(
                String, '/drone_baseline_status', default_qos)
            self.status_pub = self.create_publisher(String, '~/status', default_qos)

            self.create_service(Trigger, '~/snapshot_baseline', self.snapshot_srv)
            self.create_service(Trigger, '~/reset_baseline', self.reset_srv)
            self.create_timer(float(g('status_period')), self.publish_status)

            self._Msgs = (OccupancyGrid, Header, String, Marker, MarkerArray,
                          Pose, Point, Quaternion)
            self.get_logger().info(
                f'Change detector ready: drop >= {self.core.height_drop} m, '
                f'canopy >= {self.core.canopy_threshold} m, '
                f'persistence {self.core.persistence}, '
                f'min cluster {self.core.min_cluster_cells} cells, '
                f'baseline trigger = {self.trigger}'
                + (f', loading {self.baseline_file}' if self.load_baseline else ''))

        # -- helpers ------------------------------------------------------

        def _cell_to_world(self, row, col):
            w, h, res, ox, oy = self.geom
            return ox + (col + 0.5) * res, oy + (row + 0.5) * res

        def _bucket(self, x, y):
            return (int(math.floor(x / self.bucket_m)),
                    int(math.floor(y / self.bucket_m)))

        def _region_coverage(self, heights):
            w, h, res, ox, oy = self.geom
            x0, x1, y0, y1 = self.region
            c0 = max(0, int((x0 - ox) / res)); c1 = min(w, int((x1 - ox) / res))
            r0 = max(0, int((y0 - oy) / res)); r1 = min(h, int((y1 - oy) / res))
            if c1 <= c0 or r1 <= r0:
                return 0.0
            sub = heights[r0:r1, c0:c1]
            return float(np.count_nonzero(np.isfinite(sub))) / sub.size

        # -- baseline -----------------------------------------------------

        def _freeze(self, reason):
            if not self.core.freeze():
                self.get_logger().warn('Baseline requested but no map received yet')
                return False
            self._after_baseline(reason)
            if self.save_baseline:
                self._save()
            return True

        def _after_baseline(self, reason):
            self._alerted_lost.clear()
            self._alerted_gained.clear()
            n = self.core.baseline_canopy_cells()
            self.get_logger().info(
                f'BASELINE FROZEN ({reason}): {n} canopy cells, '
                f'region coverage {self.coverage * 100:.0f}%. Monitoring for loss.')
            self.baseline_pub.publish(self._Msgs[2](data=f'baseline_established:{n}'))

        def _save(self):
            try:
                os.makedirs(os.path.dirname(self.baseline_file) or '.', exist_ok=True)
                w, h, res, ox, oy = self.geom
                np.savez_compressed(self.baseline_file, heights=self.core.baseline,
                                    geom=np.array([w, h, res, ox, oy], dtype=np.float64),
                                    frame=np.array(self.frame_id))
                self.get_logger().info(f'Baseline saved to {self.baseline_file}')
            except OSError as exc:
                self.get_logger().error(f'Could not save baseline: {exc}')

        def _try_load(self):
            """Load the saved baseline once the live map geometry is known."""
            self._pending_load = False
            try:
                data = np.load(self.baseline_file)
                w, h, res, ox, oy = data['geom'].tolist()
                geom = (int(w), int(h), float(res), float(ox), float(oy))
            except (OSError, KeyError, ValueError) as exc:
                self.get_logger().error(
                    f'Could not load {self.baseline_file} ({exc}); building a new baseline')
                return
            if not (geom[:2] == self.geom[:2]
                    and np.allclose(geom[2:], self.geom[2:], atol=1e-6)):
                self.get_logger().error(
                    f'Saved baseline geometry {geom} does not match live map '
                    f'{self.geom}; building a new baseline instead')
                return
            self.core.set_baseline(data['heights'])
            self._after_baseline(f'loaded {self.baseline_file}')

        def survey_cb(self, msg):
            if self.trigger != 'survey' or self.core.has_baseline:
                return
            parsed = parse_survey_status(msg.data)
            if parsed is None:
                return
            wp, total, cov = parsed
            wrapped = (self._last_wp is not None and self._last_wp >= total - 1
                       and wp < self._last_wp)
            self._last_wp = wp
            if cov >= self.survey_pct or wrapped:
                if self.coverage < self.min_baseline_coverage:
                    self.get_logger().warn(
                        f'Survey pass complete but the canopy map covers only '
                        f'{self.coverage * 100:.0f}% of the survey region (need '
                        f'{self.min_baseline_coverage * 100:.0f}%). Is /parrot1/scan '
                        f'reaching scan_mapper? Not freezing; waiting for the next pass.')
                    self._last_wp = None
                    return
                self._freeze('survey pass complete' + (' (patrol looped)' if wrapped
                                                       else f', coverage={cov:.0f}%'))

        def snapshot_srv(self, _req, resp):
            resp.success = self._freeze('manual service call')
            resp.message = ('Baseline frozen' if resp.success
                            else 'No canopy map received yet')
            return resp

        def reset_srv(self, _req, resp):
            self.core.reset()
            self._last_wp = None
            self._alerted_lost.clear()
            self._alerted_gained.clear()
            self._publish_empty_change()
            self._clear_markers()
            resp.success = True
            resp.message = 'Baseline cleared; accumulating a new one'
            self.get_logger().info(resp.message)
            return resp

        # -- main callback ----------------------------------------------

        def map_cb(self, msg):
            info = msg.info
            geom = (int(info.width), int(info.height), float(info.resolution),
                    float(info.origin.position.x), float(info.origin.position.y))
            if self.geom is not None and geom != self.geom:
                self.get_logger().warn(
                    f'Canopy map geometry changed {self.geom} -> {geom}; resetting baseline')
                self.core.reset()
            self.geom = geom
            self.core.gap_bridge_cells = int(math.ceil(self.gap_bridge_m / geom[2]))
            self.frame_id = msg.header.frame_id
            self.last_info = info
            self.updates += 1

            raw = np.asarray(msg.data, dtype=np.int16).reshape(geom[1], geom[0])
            heights = np.where(raw >= 0, raw / self.height_scale, np.nan).astype(np.float32)
            self.coverage = self._region_coverage(heights)

            if self._pending_load:
                self._try_load()

            if not self.core.has_baseline:
                self.core.accumulate(heights)
                if self.trigger == 'coverage' and self.coverage >= self.coverage_fraction:
                    self._freeze(f'region coverage {self.coverage * 100:.0f}%')
                return

            lost, gained, loss_pct, gain_pct = self.core.update(heights)
            if not self.publish_gained:
                gained = np.zeros_like(gained)
            drop = self.core.baseline - heights
            lost_r, lost_rp, lost_cl = self.core.regions(lost, loss_pct, drop)
            gained_r, gained_rp, gained_cl = self.core.regions(gained, gain_pct, -drop)
            # Only accepted regions go into the change map, so pattern_scanner
            # sees solid clearings and never flags speckle.
            self._publish_change(msg.header.stamp, lost_r, gained_r & ~lost_r,
                                 lost_rp, gained_rp)
            self._n_lost_clusters = len(lost_cl)
            self._n_gained_clusters = len(gained_cl)
            self._lost_cells = int(np.count_nonzero(lost))
            self._alert(lost_cl, gained_cl)
            self._publish_markers(msg.header.stamp, lost_cl, gained_cl)

        # -- publishing ---------------------------------------------------

        def _publish_change(self, stamp, lost, gained, loss_pct, gain_pct):
            OccupancyGrid, Header = self._Msgs[0], self._Msgs[1]
            grid = np.zeros(lost.shape, dtype=np.int8)
            grid[lost] = -np.clip(np.rint(loss_pct[lost]), 1, 100).astype(np.int8)
            grid[gained] = np.clip(np.rint(gain_pct[gained]), 1, 100).astype(np.int8)
            out = OccupancyGrid()
            out.header = Header(stamp=stamp, frame_id=self.frame_id)
            out.info = self.last_info
            out.data = grid.flatten().tolist()   # already row-major (iy, ix)
            self.change_pub.publish(out)

        def _publish_empty_change(self):
            if self.geom is None:
                return
            OccupancyGrid, Header = self._Msgs[0], self._Msgs[1]
            out = OccupancyGrid()
            out.header = Header(stamp=self.get_clock().now().to_msg(),
                                frame_id=self.frame_id)
            out.info = self.last_info
            out.data = [0] * (self.geom[0] * self.geom[1])
            self.change_pub.publish(out)

        def _alert(self, lost_cl, gained_cl):
            res = self.geom[2]
            alerts = []
            for kind, clusters, seen in (('CANOPY LOST', lost_cl, self._alerted_lost),
                                         ('NEW CANOPY', gained_cl, self._alerted_gained)):
                for cl in clusters:
                    x, y = self._cell_to_world(cl['row'], cl['col'])
                    b = self._bucket(x, y)
                    if b in seen:
                        continue
                    seen.add(b)
                    area = cl['region_cells'] * res * res
                    text = (f"{kind}: ~{cl['region_cells']} cells ({area:.0f}m²) "
                            f"near ({x:.1f}, {y:.1f}), "
                            f"{'height lost' if kind == 'CANOPY LOST' else 'height gained'} "
                            f"{cl.get('mean_change_m', 0.0):.1f} m avg "
                            f"({cl['mean_pct']:.0f}%)")
                    alerts.append(text)
                    if kind == 'CANOPY LOST':
                        self.get_logger().warn(text)
                    else:
                        self.get_logger().info(text)
            if alerts:
                self.events_pub.publish(self._Msgs[2](data='; '.join(alerts)))

        def _clear_markers(self):
            Marker, MarkerArray = self._Msgs[3], self._Msgs[4]
            m = Marker()
            m.action = Marker.DELETEALL
            self.marker_pub.publish(MarkerArray(markers=[m]))

        def _publish_markers(self, stamp, lost_cl, gained_cl):
            _, Header, _, Marker, MarkerArray, Pose, Point, Quaternion = self._Msgs
            arr = MarkerArray()
            clear = Marker()
            clear.action = Marker.DELETEALL
            arr.markers.append(clear)
            res = self.geom[2]
            mid = 0
            for ns, clusters, rgb in (('canopy_lost', lost_cl[:50], (1.0, 0.0, 0.0)),
                                      ('canopy_gained', gained_cl[:50], (0.0, 0.5, 1.0))):
                for cl in clusters:
                    x, y = self._cell_to_world(cl['row'], cl['col'])
                    m = Marker()
                    m.header = Header(stamp=stamp, frame_id=self.frame_id)
                    m.ns, m.id = ns, mid
                    mid += 1
                    m.type, m.action = Marker.CYLINDER, Marker.ADD
                    m.pose = Pose(position=Point(x=x, y=y, z=5.0),
                                  orientation=Quaternion(w=1.0))
                    diameter = max(2.0 * math.sqrt(cl['region_cells'] * res * res / math.pi), 1.5)
                    m.scale.x = m.scale.y = diameter
                    m.scale.z = 10.0
                    m.color.r, m.color.g, m.color.b = rgb
                    m.color.a = 0.8
                    arr.markers.append(m)
            self.marker_pub.publish(arr)

        def publish_status(self):
            if self.geom is None:
                text = 'WAITING for /forest_canopy_map'
            elif not self.core.has_baseline:
                text = (f'BASELINE trigger={self.trigger} '
                        f'region_coverage={self.coverage * 100:.0f}%')
            else:
                text = (f'MONITORING baseline_canopy={self.core.baseline_canopy_cells()} '
                        f'lost_cells={self._lost_cells} '
                        f'lost_clusters={self._n_lost_clusters} '
                        f'gained_clusters={self._n_gained_clusters}')
            self.status_pub.publish(self._Msgs[2](data=text))

        def destroy_node(self):
            self.get_logger().info(
                f'Change detector shutting down after {self.updates} map updates.')
            super().destroy_node()

    return ChangeDetector()


if __name__ == '__main__':
    main()
