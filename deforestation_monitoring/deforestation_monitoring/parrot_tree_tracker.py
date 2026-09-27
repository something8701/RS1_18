#!/usr/bin/env python3
"""Parrot tree tracker: finds individual trees and tracks them over time.

The canopy height model (CHM) comes from the scan_mapper canopy map
(``/forest_canopy_map``). Cells the map has not scanned are filled from the
drone depth cloud (``/drone_terrain``). Trees are found with the detector in
:mod:`tree_detection`: treetops from a height-dependent window, crowns from a
saddle-aware watershed, so closely spaced trees stay separate.

Detector parameters are read from ``config/tree_detection_params.yaml``.
"""

import math
import re

import numpy as np
import rclpy
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rcl_interfaces.msg import SetParametersResult
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header, String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Pose, Quaternion
from sensor_msgs.msg import PointCloud2, PointField

from deforestation_interfaces.msg import TreeChangeEvent, TreeStatus

from .tree_detection import DetectionParams, detect_trees


class ParrotTreeTracker(Node):
    """Detect individual trees from the fused CHM and track them over time."""

    def __init__(self):
        super().__init__('parrot_tree_tracker')

        # Topics and tracker parameters
        self.declare_parameter('canopy_map_topic', '/forest_canopy_map')
        self.declare_parameter('change_map_topic', '/canopy_change_map')
        self.declare_parameter('baseline_status_topic', '/drone_baseline_status')
        self.declare_parameter('drone_terrain_topic', '/drone_terrain')
        self.declare_parameter('map_size_x', 80.0)
        self.declare_parameter('map_size_y', 80.0)
        self.declare_parameter('height_scale', 10.0)
        self.declare_parameter('min_baseline_cells', 150)
        self.declare_parameter('min_baseline_trees', 2)
        self.declare_parameter('baseline_coverage', 0.70)
        self.declare_parameter('baseline_stable_ticks', 3)
        # Coverage has plateaued when it grows by less than plateau_delta
        # (a 0-1 fraction, so 0.003 = 0.3 %) per tick for plateau_ticks ticks.
        self.declare_parameter('plateau_delta', 0.003)
        self.declare_parameter('plateau_ticks', 25)
        # After the plateau, wait up to this many ticks for trees that are
        # seen but not yet confirmed before freezing without them.
        self.declare_parameter('freeze_max_wait_ticks', 10)
        # Fallback match radius. Keep it under half the closest tree spacing
        # (~1.4 m in dense_forest) so neighbours cannot swap IDs.
        self.declare_parameter('track_radius', 0.6)
        self.declare_parameter('terrain_min_hits', 2)
        self.declare_parameter('min_gain_area_m2', 0.5)
        self.declare_parameter('gain_stable_ticks', 3)
        # A new detection within gain_crown_exclusion * crown radius of a
        # baseline tree is treated as the same tree, not a gain.
        self.declare_parameter('gain_crown_exclusion', 1.0)
        self.declare_parameter('lost_streak_threshold', 3)
        # Canopy-loss cells needed near a tree before it can be LOST. In the
        # dense removal tests removed oaks had 13-150, standing trees <= 6.
        self.declare_parameter('lost_evidence_cells', 10)
        # New-canopy cells needed near a detection before it can be GAINED,
        # so a tree that only flickers back into detection is not a gain.
        self.declare_parameter('gain_evidence_cells', 10)
        # Change cells are counted within the crown radius, capped at this
        # value so a very large crown cannot pick up unrelated cells. 2 m
        # caught every removed oak in the recorded tests with almost no
        # cells at standing trees.
        self.declare_parameter('evidence_max_radius', 2.0)
        self.declare_parameter('survey_x_min', -40.0)
        self.declare_parameter('survey_x_max', 40.0)
        self.declare_parameter('survey_y_min', -40.0)
        self.declare_parameter('survey_y_max', 40.0)
        self.declare_parameter('publish_rate', 1.0)

        # Tree detector parameters (from tree_detection_params.yaml)
        defaults = DetectionParams()
        for key, value in defaults.to_dict().items():
            self.declare_parameter(key, value)

        self.canopy_topic = self.get_parameter('canopy_map_topic').value
        self.change_topic = self.get_parameter('change_map_topic').value
        self.baseline_topic = self.get_parameter('baseline_status_topic').value
        self.terrain_topic = self.get_parameter('drone_terrain_topic').value
        self.map_x = float(self.get_parameter('map_size_x').value)
        self.map_y = float(self.get_parameter('map_size_y').value)
        self.height_scale = float(self.get_parameter('height_scale').value)
        self.min_baseline_cells = int(self.get_parameter('min_baseline_cells').value)
        self.min_baseline_trees = int(self.get_parameter('min_baseline_trees').value)
        self.baseline_coverage = float(self.get_parameter('baseline_coverage').value)
        self.baseline_stable_ticks = int(
            self.get_parameter('baseline_stable_ticks').value)
        self.plateau_delta = float(self.get_parameter('plateau_delta').value)
        self.plateau_ticks = int(self.get_parameter('plateau_ticks').value)
        self.freeze_max_wait_ticks = int(
            self.get_parameter('freeze_max_wait_ticks').value)
        self.track_radius = float(self.get_parameter('track_radius').value)
        self.terrain_min_hits = int(self.get_parameter('terrain_min_hits').value)
        self.min_gain_area_m2 = float(self.get_parameter('min_gain_area_m2').value)
        self.gain_stable_ticks = int(self.get_parameter('gain_stable_ticks').value)
        self.gain_crown_exclusion = float(
            self.get_parameter('gain_crown_exclusion').value)
        self.lost_streak_threshold = int(
            self.get_parameter('lost_streak_threshold').value)
        self.lost_evidence_cells = int(
            self.get_parameter('lost_evidence_cells').value)
        self.gain_evidence_cells = int(
            self.get_parameter('gain_evidence_cells').value)
        self.evidence_max_radius = float(
            self.get_parameter('evidence_max_radius').value)
        self.survey_x_min = float(self.get_parameter('survey_x_min').value)
        self.survey_x_max = float(self.get_parameter('survey_x_max').value)
        self.survey_y_min = float(self.get_parameter('survey_y_min').value)
        self.survey_y_max = float(self.get_parameter('survey_y_max').value)

        params_dict = defaults.to_dict()
        for key in params_dict:
            params_dict[key] = self.get_parameter(key).value
        self.itd_params = DetectionParams.from_dict(params_dict)
        self.add_on_set_parameters_callback(self._on_params_changed)

        self.res = float(self.itd_params.chm_resolution)
        self.dim_x = int(self.map_x / self.res)
        self.dim_y = int(self.map_y / self.res)
        self.origin_x = -self.map_x / 2.0
        self.origin_y = -self.map_y / 2.0

        # Accumulators
        self.terrain_height = np.zeros((self.dim_x, self.dim_y), dtype=np.float32)
        self.terrain_hits = np.zeros((self.dim_x, self.dim_y), dtype=np.int32)
        self.terrain_points = 0

        self.canopy_data = None     # [height, width] raw map values, -1 unscanned
        self.canopy_res = None
        self.canopy_ox = None
        self.canopy_oy = None

        self.change_lost = None     # bool grid of lost cells from the change map
        self.change_gained = None   # bool grid of new-canopy cells
        self.change_res = None
        self.change_ox = None
        self.change_oy = None

        # Tracking state
        self.baseline_trees = []    # [{id, x, y, height, radius_m, area_m2}]
        self.baseline_frozen = False
        self.baseline_tree_count = 0
        self.baseline_scanned = None
        self.drone_baseline_ready = False
        self.coverage_pct = 0.0
        self.next_tree_id = 1
        self.total_trees_lost = 0
        self.total_trees_gained = 0
        self._alerted_lost_ids = set()
        self._unmatched_streaks = {}
        self._tick = 0
        self._pre_freeze_ticks = 0
        self._gains_skipped_unobserved = 0
        self._gains_rejected_in_crown = 0
        self._gains_rejected_no_evidence = 0
        self._pending_gains = []
        self._freeze_candidates = []
        self._plateau_reached = False
        self._ticks_since_plateau = 0
        self._plateau_time = None
        self._disp_pre = []
        self._disp_post = []
        self._last_cov = 0.0
        self._cov_stable_ticks = 0

        # QoS
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        default_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # Subscribers
        self.canopy_sub = self.create_subscription(
            OccupancyGrid, self.canopy_topic, self.canopy_cb, map_qos)
        self.change_sub = self.create_subscription(
            OccupancyGrid, self.change_topic, self.change_cb, map_qos)
        self.baseline_sub = self.create_subscription(
            String, self.baseline_topic, self.baseline_cb, default_qos)
        self.coverage_sub = self.create_subscription(
            String, '/scan_coverage', self.coverage_cb, default_qos)
        self.terrain_sub = self.create_subscription(
            PointCloud2, self.terrain_topic, self.terrain_cb, sensor_qos)

        # Publishers
        self.status_pub = self.create_publisher(
            TreeStatus, '/parrot_tree_status', default_qos)
        self.event_pub = self.create_publisher(
            TreeChangeEvent, '/parrot_tree_change_events', default_qos)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/parrot_tree_change_markers', default_qos)
        self.positions_pub = self.create_publisher(
            PointCloud2, '/parrot_tree_positions', default_qos)
        self.treetop_pub = self.create_publisher(
            MarkerArray, '/parrot_treetop_markers', default_qos)
        self.crown_pub = self.create_publisher(
            OccupancyGrid, '/parrot_crown_map', map_qos)
        # Frozen baseline as a latched cloud (x, y, z = tree id).
        self.baseline_pub = self.create_publisher(
            PointCloud2, '/parrot_tree_baseline', map_qos)
        self.drone_baseline_pub = self.create_publisher(
            String, '/drone_baseline_status', default_qos)

        self.reset_srv = self.create_service(
            Trigger, '~/reset_baseline', self.reset_baseline_callback)

        period = 1.0 / max(0.5, float(self.get_parameter('publish_rate').value))
        self.create_timer(period, self.publish_status)

        self.get_logger().info(
            f'Parrot Tree Tracker ready (CHM ITD). Grid: {self.dim_x}x{self.dim_y} '
            f'at {self.res}m. window_scale={self.itd_params.window_scale}, '
            f'saddle_ratio={self.itd_params.saddle_ratio}, '
            f'smooth_sigma={self.itd_params.smooth_sigma}.'
        )

    def _on_params_changed(self, params):
        """Apply parameter changes at runtime, without restarting the node."""
        fields = set(self.itd_params.to_dict().keys())
        for param in params:
            if param.name == 'chm_resolution':
                continue  # the accumulator grid size is fixed at startup
            if param.name == 'baseline_coverage':
                self.baseline_coverage = float(param.value)
                continue
            if param.name == 'baseline_stable_ticks':
                self.baseline_stable_ticks = int(param.value)
                continue
            if param.name == 'plateau_delta':
                self.plateau_delta = float(param.value)
                continue
            if param.name == 'plateau_ticks':
                self.plateau_ticks = int(param.value)
                continue
            if param.name in fields:
                setattr(self.itd_params, param.name, param.value)
        self.get_logger().info(
            f'ITD params updated: smooth={self.itd_params.smooth_sigma}, '
            f'min_height={self.itd_params.min_height}, '
            f'window_scale={self.itd_params.window_scale}, '
            f'saddle={self.itd_params.saddle_ratio}, '
            f'min_crown_cells={self.itd_params.min_crown_cells}')
        return SetParametersResult(successful=True)

    # Callbacks

    def terrain_cb(self, cloud: PointCloud2):
        """Accumulate the drone depth cloud as a max-height grid."""
        offsets = {}
        for field in cloud.fields:
            if field.name in ('x', 'y', 'z'):
                offsets[field.name] = field.offset
        if len(offsets) != 3 or cloud.point_step <= 0 or not cloud.data:
            return

        raw = np.frombuffer(bytes(cloud.data), dtype=np.uint8)
        usable = raw.size - (raw.size % cloud.point_step)
        if usable <= 0:
            return
        arr = raw[:usable].reshape(-1, cloud.point_step)
        dtype = '>f4' if cloud.is_bigendian else '<f4'

        def col(name):
            off = offsets[name]
            return arr[:, off:off + 4].copy().view(dtype).ravel()

        xs, ys, zs = col('x'), col('y'), col('z')
        valid = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
        xs, ys, zs = xs[valid], ys[valid], zs[valid]
        if len(xs) == 0:
            return

        ix = ((xs - self.origin_x) / self.res).astype(np.int64)
        iy = ((ys - self.origin_y) / self.res).astype(np.int64)
        inside = (ix >= 0) & (ix < self.dim_x) & (iy >= 0) & (iy < self.dim_y)
        ix, iy, zs = ix[inside], iy[inside], zs[inside]
        if len(ix) == 0:
            return
        np.maximum.at(self.terrain_height, (ix, iy), zs)
        np.add.at(self.terrain_hits, (ix, iy), 1)
        self.terrain_points += len(ix)

    def canopy_cb(self, grid: OccupancyGrid):
        data = np.array(grid.data, dtype=np.float32).reshape(
            grid.info.height, grid.info.width)
        self.canopy_data = data
        self.canopy_res = grid.info.resolution
        self.canopy_ox = grid.info.origin.position.x
        self.canopy_oy = grid.info.origin.position.y

    def change_cb(self, grid: OccupancyGrid):
        data = np.array(grid.data, dtype=np.int8).reshape(
            grid.info.height, grid.info.width)
        self.change_lost = data <= -50
        self.change_gained = data >= 50
        self.change_res = grid.info.resolution
        self.change_ox = grid.info.origin.position.x
        self.change_oy = grid.info.origin.position.y

    def baseline_cb(self, msg: String):
        if not self.drone_baseline_ready:
            self.drone_baseline_ready = True
            self.get_logger().info(f'Drone canopy baseline received: {msg.data}')

    def coverage_cb(self, msg: String):
        match = re.search(r'coverage=([0-9.]+)%', msg.data)
        if match:
            self.coverage_pct = float(match.group(1)) / 100.0

    # CHM fusion

    def _terrain_ground_offset(self) -> float:
        """Estimate the height offset of the terrain cloud's ground.

        Because the depth camera is tilted, the cloud's ground is not at 0 m
        (usually 0.3-5 m). Where the canopy map shows ground (< 1 m), the
        terrain height there is the offset. Without that overlap, use a low
        percentile of all terrain heights.
        """
        seen = self.terrain_hits > 0
        if not np.any(seen):
            return 0.0
        if self.canopy_data is not None:
            map_ground = (self.canopy_data >= 0) & (self.canopy_data < self.height_scale)
            rows, cols = np.nonzero(map_ground)
            if len(rows) >= 20:
                wx = self.canopy_ox + (cols + 0.5) * self.canopy_res
                wy = self.canopy_oy + (rows + 0.5) * self.canopy_res
                ix = ((wx - self.origin_x) / self.res).astype(np.int64)
                iy = ((wy - self.origin_y) / self.res).astype(np.int64)
                ok = (ix >= 0) & (ix < self.dim_x) & (iy >= 0) & (iy < self.dim_y)
                ix, iy = ix[ok], iy[ok]
                samples = self.terrain_height[ix, iy][seen[ix, iy]]
                if samples.size >= 20:
                    return float(np.percentile(samples, 25.0))
        heights = self.terrain_height[seen]
        return float(np.percentile(heights, 10.0))

    def _fused_chm(self):
        """Combine the canopy map and terrain cloud into (chm, scanned).

        The canopy map is already height above ground, so it is used first.
        The terrain cloud fills cells the map has not scanned, minus its
        ground offset.
        """
        fused = np.zeros((self.dim_x, self.dim_y), dtype=np.float32)
        scanned = np.zeros((self.dim_x, self.dim_y), dtype=bool)

        # 1. Canopy map.
        if self.canopy_data is not None and self.canopy_res is not None:
            # Sample the map at every CHM cell centre. Mapping each map
            # cell forward to one CHM cell leaves stripes of holes when the
            # two resolutions differ.
            fx = self.origin_x + (np.arange(self.dim_x) + 0.5) * self.res
            fy = self.origin_y + (np.arange(self.dim_y) + 0.5) * self.res
            cols = np.floor((fx - self.canopy_ox) / self.canopy_res).astype(np.int64)
            rows = np.floor((fy - self.canopy_oy) / self.canopy_res).astype(np.int64)
            h, w = self.canopy_data.shape
            ok_x = (cols >= 0) & (cols < w)
            ok_y = (rows >= 0) & (rows < h)
            ix = np.nonzero(ok_x)[0]
            iy = np.nonzero(ok_y)[0]
            if len(ix) and len(iy):
                # canopy_data is [row=y, col=x]; the CHM is [ix, iy].
                sampled = self.canopy_data[np.ix_(rows[iy], cols[ix])].T
                seen = sampled >= 0
                block = np.where(seen, sampled / self.height_scale, 0.0)
                fused[np.ix_(ix, iy)] = block
                scanned[np.ix_(ix, iy)] = seen

        # 2. Terrain cloud fills the remaining holes. Cells seen fewer than
        # terrain_min_hits times are mostly single grazing beams, so they
        # stay unknown.
        terrain_seen = self.terrain_hits >= self.terrain_min_hits
        if np.any(terrain_seen):
            offset = self._terrain_ground_offset()
            fill = terrain_seen & ~scanned
            fused[fill] = np.clip(self.terrain_height[fill] - offset, 0.0, None)
            scanned[fill] = True
        return fused, scanned

    # Baseline / change handling

    def _crown_window(self, det, grid):
        """Return (window, circle, r_px) for a crown: the square of ``grid``
        around it and a mask of the cells inside its crown circle."""
        r_px = max(1, int(round(det.radius_m / self.res)))
        ix = int((det.x - self.origin_x) / self.res)
        iy = int((det.y - self.origin_y) / self.res)
        x0, x1 = max(0, ix - r_px), min(self.dim_x, ix + r_px + 1)
        y0, y1 = max(0, iy - r_px), min(self.dim_y, iy + r_px + 1)
        window = grid[x0:x1, y0:y1]
        # The grid is [ix, iy]: rows are x, columns are y.
        rows, cols = np.mgrid[0:window.shape[0], 0:window.shape[1]]
        circle = (rows - (ix - x0)) ** 2 + (cols - (iy - y0)) ** 2 <= r_px ** 2
        return window, circle, r_px

    def _neighborhood_observed_fraction(self, det, scanned) -> float:
        """Fraction of the crown circle the drone has observed.

        Thin stripes of missing returns inside a crown are scan gaps, so the
        scanned mask is closed (gap-filled) before counting. The question is
        whether the drone flew over the crown, not whether every cell returned.
        """
        window, circle, _ = self._crown_window(det, scanned)
        if window.size == 0:
            return 0.0
        # binary_closing treats the area outside the array as unobserved and
        # erodes the window border, where the crown circle touches it. Pad
        # with edge values and crop back afterwards.
        iterations = max(1, int(self.itd_params.mask_close_iterations))
        padded = np.pad(window, iterations, mode='edge')
        closed = ndimage.binary_closing(
            padded,
            structure=np.ones((3, 3), dtype=bool),
            iterations=iterations)
        window = closed[iterations:-iterations, iterations:-iterations]
        return float(window[circle].mean()) if np.any(circle) else 0.0

    def _crown_coverage_detail(self, det, scanned):
        """Position, radius (m and cells), cells checked/observed, fraction."""
        window, circle, r_px = self._crown_window(det, scanned)
        gap_fraction = self._neighborhood_observed_fraction(det, scanned)
        return {
            'x': det.x, 'y': det.y,
            'radius_m': det.radius_m,
            'radius_cells': r_px,
            # Raw counts inside the crown circle. 'fraction' is the
            # gap-filled value that admission uses.
            'checked': int(circle.sum()),
            'observed': int(np.count_nonzero(window[circle])),
            'fraction': gap_fraction,
        }

    def _freeze_baseline(self, detections, scanned=None):
        # A tree enters the baseline only with the same checks as a gain:
        # minimum crown area and at least 80% of its crown observed. This
        # keeps short-lived fragments out, since they would later go "lost".
        eligible = []
        for d in detections:
            area_ok = d.area_m2 >= self.min_gain_area_m2
            if scanned is None:
                if area_ok:
                    eligible.append(d)
                continue
            detail = self._crown_coverage_detail(d, scanned)
            observed_ok = detail['fraction'] >= 0.8
            if area_ok and observed_ok:
                eligible.append(d)
            reasons = []
            if not area_ok:
                reasons.append(f'area<{self.min_gain_area_m2}')
            if not observed_ok:
                reasons.append('observed<0.8')
            self.get_logger().info(
                f'crown coverage ({d.x:.1f},{d.y:.1f}): '
                f'area={d.area_m2:.2f}m2 '
                f'{"REJECT " + ",".join(reasons) if reasons else "ADMIT"} '
                f'r={d.radius_m:.2f}m ({detail["radius_cells"]} cells), '
                f'checked={detail["checked"]}, observed={detail["observed"]}, '
                f'fraction={detail["fraction"]:.2f}')
            if not observed_ok:
                sub, _, _ = self._crown_window(d, scanned)
                step = max(1, sub.shape[0] // 24)
                rows = [''.join('#' if sub[i, j] else '.'
                                for j in range(0, sub.shape[1], step))
                        for i in range(0, sub.shape[0], step)]
                self.get_logger().info(
                    f'crown hit pattern (raw, {step}-cell bins):\n'
                    + '\n'.join(rows))
        if len(eligible) < max(1, self.min_baseline_trees):
            self.baseline_frozen = False
            self.get_logger().info(
                f'Baseline blocked: {len(eligible)} of {len(detections)} pass '
                f'admission, need {self.min_baseline_trees}',
                throttle_duration_sec=5.0)
            return False, (
                f'blocked: crown coverage, '
                f'{len(eligible)}/{len(detections)} candidates >=80% observed '
                f'(need {self.min_baseline_trees})')
        self.baseline_trees = [{
            'id': i,
            'x': d.x, 'y': d.y, 'height': d.height,
            'radius_m': d.radius_m, 'area_m2': d.area_m2,
        } for i, d in enumerate(eligible, start=1)]
        self.next_tree_id = len(self.baseline_trees) + 1
        self.baseline_tree_count = len(self.baseline_trees)
        self.baseline_scanned = (scanned.copy()
                                 if scanned is not None else None)
        self.baseline_frozen = True
        self._alerted_lost_ids.clear()
        self._unmatched_streaks.clear()
        self.total_trees_lost = 0
        self.total_trees_gained = 0
        if self._plateau_time is not None:
            now = self.get_clock().now().nanoseconds / 1e9
            self.get_logger().info(
                f'plateau-to-freeze: {max(0.0, now - self._plateau_time):.1f}s')
        self.get_logger().info(
            f'Parrot tree baseline frozen: {len(self.baseline_trees)} '
            f'individual trees ({len(detections) - len(eligible)} rejected)')
        self.drone_baseline_pub.publish(String(
            data=f'frozen: {len(self.baseline_trees)} trees'))
        self._publish_baseline()
        return True, ''

    def reset_baseline_callback(self, request, response):
        fused, scanned = self._fused_chm()
        if not np.any(scanned):
            response.success = False
            response.message = 'No CHM data yet; cannot re-baseline.'
            return response
        detections, _, _ = detect_trees(
            fused, scanned, self.itd_params, self.origin_x, self.origin_y,
            return_debug=True)
        ok, reason = self._freeze_baseline(detections, scanned)
        if not ok:
            response.success = False
            response.message = reason
            return response
        self.total_trees_lost = 0
        self.total_trees_gained = 0
        response.success = True
        response.message = (
            f'Baseline reset. {len(self.baseline_trees)} individual trees.')
        return response

    def _match_baseline(self, detections):
        """Match detections to baseline IDs one-to-one.

        Each tree's match radius is ``min(0.4 * nearest neighbour distance,
        1.5)`` m, so trees 1.4 m apart cannot swap IDs. The Hungarian
        algorithm minimises the total distance over all pairs.

        Returns ``(matched_ids, unmatched_base_indices)``. ``matched_ids[i]``
        is the baseline ID of detection i, or None for a new tree.
        """
        ids = [None] * len(detections)
        if not self.baseline_trees or not detections:
            return ids, list(range(len(self.baseline_trees)))

        base = np.array([[t['x'], t['y']] for t in self.baseline_trees])
        det = np.array([[d.x, d.y] for d in detections])
        d_bb = np.hypot(
            base[:, None, 0] - base[None, :, 0],
            base[:, None, 1] - base[None, :, 1])
        np.fill_diagonal(d_bb, np.inf)
        nearest = d_bb.min(axis=1)
        radius = np.minimum(0.4 * nearest, 1.5)
        # A lone tree has no neighbour; use the configured fallback radius.
        radius = np.where(np.isinf(nearest), self.track_radius, radius)

        cost = np.hypot(
            det[:, None, 0] - base[None, :, 0],
            det[:, None, 1] - base[None, :, 1])
        allowed = cost <= radius[None, :]
        big = 1e6
        matrix = np.where(allowed, cost, big)
        rows, cols = linear_sum_assignment(matrix)

        unmatched = list(range(len(self.baseline_trees)))
        for ri, ci in zip(rows, cols):
            if matrix[ri, ci] < big:
                ids[ri] = self.baseline_trees[ci]['id']
                unmatched.remove(ci)
        return ids, unmatched

    def _containing_crown(self, det):
        """Return the baseline tree whose crown contains det, else None."""
        best, best_d = None, float('inf')
        for tree in self.baseline_trees:
            if tree['id'] in self._alerted_lost_ids:
                continue  # a crown already reported lost can be regrown
            d = math.hypot(det.x - tree['x'], det.y - tree['y'])
            if d <= self.gain_crown_exclusion * tree['radius_m'] and d < best_d:
                best, best_d = tree, d
        return best

    def _lost_cells_near(self, tree, radius) -> int:
        """Count change-map lost cells within radius of the tree."""
        return self._change_cells_near(self.change_lost, tree, radius)

    def _gained_cells_near(self, tree, radius) -> int:
        """Count change-map new-canopy cells within radius of the tree."""
        return self._change_cells_near(self.change_gained, tree, radius)

    def _evidence_radius(self, radius_m) -> float:
        return max(self.track_radius, min(radius_m, self.evidence_max_radius))

    def _change_cells_near(self, mask, tree, radius) -> int:
        if mask is None or not np.any(mask):
            return 0
        rows, cols = np.nonzero(mask)
        wx = self.change_ox + (cols + 0.5) * self.change_res
        wy = self.change_oy + (rows + 0.5) * self.change_res
        close = (wx - tree['x']) ** 2 + (wy - tree['y']) ** 2 <= radius ** 2
        return int(np.count_nonzero(close))

    # Publishing

    def publish_status(self):
        fused, scanned = self._fused_chm()
        if not np.any(scanned):
            return
        detections, labels, canopy = detect_trees(
            fused, scanned, self.itd_params, self.origin_x, self.origin_y,
            return_debug=True)

        # Ignore detections outside the survey box (edge artifacts).
        detections = [
            d for d in detections
            if self.survey_x_min <= d.x <= self.survey_x_max
            and self.survey_y_min <= d.y <= self.survey_y_max
        ]

        # Freeze the tree baseline once enough canopy has been observed.
        if not self.baseline_frozen:
            covered = int(np.count_nonzero(scanned))
            # coverage_pct is a 0-1 fraction, like plateau_delta.
            if self.coverage_pct - self._last_cov < self.plateau_delta:
                self._cov_stable_ticks += 1
            else:
                self._cov_stable_ticks = 0
            self._last_cov = self.coverage_pct
            plateau = self._cov_stable_ticks >= self.plateau_ticks
            if plateau and not self._plateau_reached:
                self._plateau_reached = True
                self._plateau_time = self.get_clock().now().nanoseconds / 1e9
                # Only count admission ticks after the plateau.
                self._freeze_candidates = []
            if self._plateau_reached:
                self._ticks_since_plateau += 1

            # Same adaptive match radius as tracking:
            # min(0.4 * nearest neighbour distance, 1.5 m).
            det_pos = np.array([[d.x, d.y] for d in detections])
            if len(det_pos) > 1:
                d_nn = np.hypot(
                    det_pos[:, None, 0] - det_pos[None, :, 0],
                    det_pos[:, None, 1] - det_pos[None, :, 1])
                np.fill_diagonal(d_nn, np.inf)
                radii = np.minimum(0.4 * d_nn.min(axis=1), 1.5)
                radii = np.where(np.isinf(radii), self.track_radius, radii)
            else:
                radii = np.full(len(det_pos), self.track_radius)

            # A tree is stable when seen in 3 of the last 4 ticks, so one
            # missed detection does not reset it.
            seen = [False] * len(self._freeze_candidates)
            for det, radius in zip(detections, radii):
                best_idx, best_d = None, float('inf')
                for idx, cand in enumerate(self._freeze_candidates):
                    d = math.hypot(det.x - cand[0], det.y - cand[1])
                    if d <= cand[3] and d < best_d:
                        best_idx, best_d = idx, d
                if best_idx is None:
                    # [x, y, last-4-ticks hits, radius, misses, last det]
                    cand = [det.x, det.y, [0, 0, 0, 0], float(radius), 0, det]
                    cand[2][-1] = 1
                    self._freeze_candidates.append(cand)
                    seen.append(True)
                else:
                    cand = self._freeze_candidates[best_idx]
                    (self._disp_post if self._plateau_reached
                     else self._disp_pre).append(best_d)
                    cand[0], cand[1], cand[3] = det.x, det.y, float(radius)
                    cand[5] = det
                    cand[2].append(1)
                    cand[2] = cand[2][-4:]
                    cand[4] = 0
                    seen[best_idx] = True
            kept = []
            for i, cand in enumerate(self._freeze_candidates):
                if seen[i]:
                    kept.append(cand)
                    continue
                cand[2].append(0)
                cand[2] = cand[2][-4:]
                cand[4] += 1
                if cand[4] < 2:  # tolerate one missed tick before dropping
                    kept.append(cand)
            self._freeze_candidates = kept

            # _tick only counts after the freeze, so this log uses its own
            # counter.
            self._pre_freeze_ticks += 1
            if self._pre_freeze_ticks % 20 == 0 and \
                    (self._disp_pre or self._disp_post):
                for name, arr in (('pre-plateau', self._disp_pre),
                                  ('post-plateau', self._disp_post)):
                    if arr:
                        a = np.array(arr)
                        self.get_logger().info(
                            f'treetop displacement {name}: '
                            f'max={a.max():.2f}m p95={np.percentile(a, 95):.2f}m '
                            f'(n={len(a)})')

            # Freeze from every confirmed candidate (seen in 3 of the last 4
            # ticks) using its latest detection, not only the trees detected
            # on this tick. Otherwise a tree missed on the freeze tick is left
            # out and later reported as GAINED.
            stable_dets = [
                cand[5] for cand in self._freeze_candidates
                if sum(cand[2]) >= self.baseline_stable_ticks]
            confirmed = len(stable_dets)
            # Candidates not yet confirmed. The list is reset at the plateau,
            # so a tree that misses one of the first ticks would be frozen
            # out. Wait up to freeze_max_wait_ticks for them to confirm.
            pending = len(self._freeze_candidates) - confirmed
            waiting_for_pending = (
                pending > 0
                and self._ticks_since_plateau < self.freeze_max_wait_ticks)
            self.drone_baseline_pub.publish(String(data=(
                f'waiting: coverage {self.coverage_pct * 100:.1f}%, '
                f'plateau {self._cov_stable_ticks}/{self.plateau_ticks}, '
                f'candidates confirmed {confirmed}/{len(self._freeze_candidates)}'
                f'{f", {pending} pending" if pending else ""}, '
                f'min {self.min_baseline_trees}')))
            if self.coverage_pct >= self.baseline_coverage and \
                    covered >= self.min_baseline_cells and \
                    len(stable_dets) >= self.min_baseline_trees and \
                    self._cov_stable_ticks >= self.plateau_ticks and \
                    not waiting_for_pending:
                if pending:
                    self.get_logger().info(
                        f'freezing without {pending} unconfirmed candidate(s) '
                        f'after {self._ticks_since_plateau} ticks')
                ok, reason = self._freeze_baseline(stable_dets, scanned)
                if not ok:
                    self.drone_baseline_pub.publish(String(data=reason))
                    return
            else:
                return

        if not self.baseline_frozen:
            self._publish_positions(detections)
            return

        matched_ids, unmatched = self._match_baseline(detections)

        # Log how many baseline trees are unmatched on this tick, before the
        # lost/gained filters decide whether to report anything.
        self._tick += 1
        if self._tick % 5 == 0:
            self.get_logger().info(
                f'pre-hysteresis: {len(unmatched)}/{self.baseline_tree_count} '
                f'trees unmatched this tick')
        if self._tick % 50 == 0:
            self.get_logger().info(
                f'gains skipped (unobserved at baseline): '
                f'{self._gains_skipped_unobserved}, '
                f'rejected (inside a crown): {self._gains_rejected_in_crown}, '
                f'rejected (no new-canopy evidence): '
                f'{self._gains_rejected_no_evidence}')

        # Reset the "still missing" streak for every tree seen this tick.
        seen_ids = {tid for tid in matched_ids if tid is not None}
        for tid in seen_ids:
            self._unmatched_streaks.pop(tid, None)

        # New trees (GAINED): a detection gets an ID only after it has been
        # seen in the same place for several ticks.
        for di, tid in enumerate(matched_ids):
            if tid is not None:
                continue
            det = detections[di]
            if det.area_m2 < self.min_gain_area_m2:
                continue
            # Same treetop on consecutive ticks, allowing normal jitter.
            slot = None
            for idx, (px, py, count) in enumerate(self._pending_gains):
                if math.hypot(det.x - px, det.y - py) <= self.track_radius:
                    slot = idx
                    break
            if slot is None:
                self._pending_gains.append([det.x, det.y, 1])
                continue
            px, py, count = self._pending_gains[slot]
            self._pending_gains[slot] = [
                (px * count + det.x) / (count + 1),
                (py * count + det.y) / (count + 1),
                count + 1,
            ]
            if self._pending_gains[slot][2] < self.gain_stable_ticks:
                continue
            self._pending_gains.pop(slot)
            # A detection inside an existing crown is that tree's top moving
            # to another lobe (big oaks have several), not a new tree.
            host = self._containing_crown(det)
            if host is not None:
                self._gains_rejected_in_crown += 1
                self.get_logger().info(
                    f'gain rejected: ({det.x:.1f},{det.y:.1f}) is inside the '
                    f'crown of tree #{host["id"]} at '
                    f'({host["x"]:.1f},{host["y"]:.1f}), '
                    f'r={host["radius_m"]:.1f}m',
                    throttle_duration_sec=10.0)
                continue
            # A real gain needs its crown area to have been observed at
            # baseline. Otherwise it is just newly scanned area, which is
            # logged but not reported as a change.
            cell_ix = int((det.x - self.origin_x) / self.res)
            cell_iy = int((det.y - self.origin_y) / self.res)
            if self.baseline_scanned is not None:
                r_px = max(1, int(round(det.radius_m / self.res)))
                x0, x1 = max(0, cell_ix - r_px), min(self.dim_x, cell_ix + r_px + 1)
                y0, y1 = max(0, cell_iy - r_px), min(self.dim_y, cell_iy + r_px + 1)
                window = self.baseline_scanned[x0:x1, y0:y1]
                if window.size == 0 or window.mean() < 0.8:
                    self._gains_skipped_unobserved += 1
                    self.get_logger().info(
                        f'newly observed (not a change event): '
                        f'({det.x:.1f},{det.y:.1f})',
                        throttle_duration_sec=10.0)
                    continue
            evidence = self._gained_cells_near(
                {'x': det.x, 'y': det.y}, self._evidence_radius(det.radius_m))
            if evidence < self.gain_evidence_cells:
                self._gains_rejected_no_evidence += 1
                self.get_logger().info(
                    f'gain rejected: ({det.x:.1f},{det.y:.1f}) has '
                    f'{evidence} new-canopy cells nearby '
                    f'(need {self.gain_evidence_cells})',
                    throttle_duration_sec=10.0)
                continue
            new_id = self.next_tree_id
            self.next_tree_id += 1
            self.baseline_trees.append({
                'id': new_id,
                'x': det.x, 'y': det.y, 'height': det.height,
                'radius_m': det.radius_m, 'area_m2': det.area_m2,
            })
            matched_ids[di] = new_id
            self.total_trees_gained += 1
            self._publish_event('GAINED', new_id, det.x, det.y, det.area_m2)

        # Lost trees: the tree must be missing for several ticks and there
        # must be enough canopy-loss cells near it.
        for bi in unmatched:
            tree = self.baseline_trees[bi]
            if tree['id'] in self._alerted_lost_ids:
                continue
            radius = self._evidence_radius(tree['radius_m'])
            if self._lost_cells_near(tree, radius) < self.lost_evidence_cells:
                self._unmatched_streaks.pop(tree['id'], None)
                continue
            streak = self._unmatched_streaks.get(tree['id'], 0) + 1
            self._unmatched_streaks[tree['id']] = streak
            if streak < self.lost_streak_threshold:
                continue
            self._alerted_lost_ids.add(tree['id'])
            self.total_trees_lost += 1
            self._publish_event(
                'LOST', tree['id'], tree['x'], tree['y'], tree['area_m2'])
            self._publish_change_marker(tree, lost=True)

        status = TreeStatus()
        status.header = Header(
            stamp=self.get_clock().now().to_msg(), frame_id='parrot1_odom')
        status.baseline_count = self.baseline_tree_count - self.total_trees_lost
        status.current_count = len(seen_ids)
        status.lost_count = self.total_trees_lost
        status.gained_count = self.total_trees_gained
        self.status_pub.publish(status)

        self._publish_positions(detections)
        self._publish_treetop_markers(detections)
        self._publish_crown_map(labels, canopy, fused)

    # Message builders

    def _publish_event(self, event_type, tree_id, x, y, area_m2):
        event = TreeChangeEvent()
        event.header = Header(
            stamp=self.get_clock().now().to_msg(), frame_id='parrot1_odom')
        event.event_type = event_type
        event.tree_id = int(tree_id)
        event.x = float(x)
        event.y = float(y)
        event.area_m2 = float(area_m2)
        self.event_pub.publish(event)
        self.get_logger().info(
            f'Parrot tree event: {event_type} #{tree_id} at ({x:.1f}, {y:.1f})')

    def _publish_change_marker(self, tree, lost):
        marker = Marker()
        marker.header = Header(
            stamp=self.get_clock().now().to_msg(), frame_id='parrot1_odom')
        marker.id = tree['id']
        marker.ns = 'parrot_tree_lost' if lost else 'parrot_tree_gained'
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose = Pose(
            position=Point(x=tree['x'], y=tree['y'], z=2.5),
            orientation=Quaternion(w=1.0))
        size = max(tree['area_m2'] ** 0.5, 1.0)
        marker.scale.x = size
        marker.scale.y = size
        marker.scale.z = 5.0
        marker.color.r = 1.0 if lost else 0.0
        marker.color.g = 0.0
        marker.color.b = 0.0 if lost else 1.0
        marker.color.a = 0.9
        self.marker_pub.publish(MarkerArray(markers=[marker]))

    def _publish_baseline(self):
        pts = np.array([[t['x'], t['y'], float(t['id'])]
                        for t in self.baseline_trees], dtype=np.float32)
        self.baseline_pub.publish(self._xyz_cloud(pts.reshape(-1, 3)))

    def _xyz_cloud(self, pts):
        cloud = PointCloud2()
        cloud.header = Header(
            stamp=self.get_clock().now().to_msg(), frame_id='parrot1_odom')
        cloud.height = 1
        cloud.width = len(pts)
        cloud.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        cloud.is_bigendian = False
        cloud.point_step = 12
        cloud.row_step = cloud.point_step * cloud.width
        cloud.is_dense = True
        cloud.data = pts.astype(np.float32).tobytes()
        return cloud

    def _publish_positions(self, detections):
        pts = np.array([[d.x, d.y, 0.0] for d in detections], dtype=np.float32)
        self.positions_pub.publish(self._xyz_cloud(pts.reshape(-1, 3)))

    def _publish_treetop_markers(self, detections):
        markers = MarkerArray()
        for i, det in enumerate(detections):
            marker = Marker()
            marker.header = Header(
                stamp=self.get_clock().now().to_msg(), frame_id='parrot1_odom')
            marker.id = i
            marker.ns = 'treetop'
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose = Pose(
                position=Point(x=det.x, y=det.y, z=det.height),
                orientation=Quaternion(w=1.0))
            marker.scale.x = 0.6
            marker.scale.y = 0.6
            marker.scale.z = 0.6
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.9
            markers.markers.append(marker)
        self.treetop_pub.publish(markers)

    def _publish_crown_map(self, labels, canopy, fused):
        data = np.where(canopy, labels, 0).astype(np.int8)
        grid = OccupancyGrid()
        grid.header = Header(
            stamp=self.get_clock().now().to_msg(), frame_id='parrot1_odom')
        grid.info.resolution = self.res
        grid.info.width = self.dim_x
        grid.info.height = self.dim_y
        grid.info.origin = Pose(
            position=Point(x=self.origin_x, y=self.origin_y, z=0.0),
            orientation=Quaternion(w=1.0))
        grid.data = data.T.flatten().tolist()
        self.crown_pub.publish(grid)

    def destroy_node(self):
        self.get_logger().info(
            f'Parrot Tree Tracker shutting down. {self.terrain_points} terrain '
            f'points. Lost: {self.total_trees_lost}, Gained: {self.total_trees_gained}.'
        )
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ParrotTreeTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
