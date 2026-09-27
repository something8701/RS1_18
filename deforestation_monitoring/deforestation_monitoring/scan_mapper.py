#!/usr/bin/env python3
"""
Canopy scan mapper: builds a canopy height map from the Parrot drone's 2D
LiDAR.

The LiDAR is pitched about 90 degrees down, so each scan is a line across
the forest below (push-broom). Each beam is rotated by the LiDAR mounting
angles (pitch/yaw parameters) and the drone's orientation into the map
frame, then stored in a height-per-cell grid.

Outputs:
  - /forest_canopy_map    OccupancyGrid: canopy height 0-100 (0-10 m),
                          -1 = not scanned yet
  - /scan_coverage        String: percentage of the survey area scanned
  - /drone_lidar_points   PointCloud2: recent swath points (for RViz)
  - /canopy_change_map    OccupancyGrid: -100 = canopy lost, +100 = new
  - /canopy_change_events String alerts for connected lost regions
  - /canopy_change_markers MarkerArray (red = lost, blue = gained)
  - /drone_baseline_status String

Changes are measured against a baseline snapshot taken once enough of the
survey area has been scanned.
"""

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import numpy as np
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Point, Pose, Quaternion
from std_msgs.msg import Header, String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray
from scipy import ndimage
import math


def quaternion_matrix(q: Quaternion) -> np.ndarray:
    """3x3 rotation matrix from a geometry_msgs/Quaternion."""
    x, y, z, w = q.x, q.y, q.z, q.w
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
    ], dtype=np.float64)


def project_scan_to_world(
    ranges: np.ndarray,
    angles: np.ndarray,
    q: Quaternion,
    pos: np.ndarray,
    altitude: float,
    pitch: float = math.pi / 2.0,
    yaw: float = 0.0,
) -> np.ndarray:
    """Project one 2D laser scan into the world frame.

    Used by both the ScanMapper node and the offline replay evaluator, so
    both build the map the same way.
    """
    cp, sp = math.cos(pitch), math.sin(pitch)
    R_pitch = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    cy, sy = math.cos(yaw), math.sin(yaw)
    R_yaw = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    d = np.column_stack([np.cos(angles), np.sin(angles),
                         np.zeros(len(angles))])
    d_body = d @ R_pitch.T @ R_yaw.T
    R = quaternion_matrix(q)
    pos3 = np.array([pos[0], pos[1], altitude], dtype=np.float64)
    return (d_body * ranges[:, None]) @ R.T + pos3


class ScanMapper(Node):
    """Maps canopy height from the drone's pitched LiDAR swaths."""

    def __init__(self):
        super().__init__('scan_mapper')

        # Parameters
        self.declare_parameter('resolution', 0.5,
            descriptor=ParameterDescriptor(description='Grid cell size in metres'))
        self.declare_parameter('map_size_x', 80.0,
            descriptor=ParameterDescriptor(description='Map X dimension in metres'))
        self.declare_parameter('map_size_y', 80.0,
            descriptor=ParameterDescriptor(description='Map Y dimension in metres'))
        self.declare_parameter('scan_topic', '/parrot1/scan',
            descriptor=ParameterDescriptor(description='Drone LaserScan topic'))
        self.declare_parameter('odom_topic', '/parrot1/odometry',
            descriptor=ParameterDescriptor(description='Drone odometry topic'))
        self.declare_parameter('map_frame', 'parrot1_odom',
            descriptor=ParameterDescriptor(description='Frame for the published maps'))
        self.declare_parameter('sensor_pitch', math.pi / 2.0,
            descriptor=ParameterDescriptor(
                description='Static LiDAR pitch about the body Y axis (rad)'))
        self.declare_parameter('sensor_yaw', 0.0,
            descriptor=ParameterDescriptor(
                description='Static LiDAR yaw about the body Z axis (rad)'))
        self.declare_parameter('range_min', 0.5,
            descriptor=ParameterDescriptor(
                description='Minimum range accepted from the scan (m)'))
        self.declare_parameter('altitude', 10.0,
            descriptor=ParameterDescriptor(
                description='Drone flight altitude (m). The sim odometry always '
                'reports z=0, so this fixed value is used instead.'))
        self.declare_parameter('canopy_threshold', 2.0,
            descriptor=ParameterDescriptor(
                description='Cell height (m) above which a cell counts as canopy'))
        self.declare_parameter('height_scale', 10.0,
            descriptor=ParameterDescriptor(
                description='Map value = min(height * height_scale, 100); 10 means 0-10 m'))
        self.declare_parameter('baseline_threshold', 100,
            descriptor=ParameterDescriptor(
                description='Minimum scans before the baseline snapshot may trigger'))
        self.declare_parameter('coverage_required', 0.8,
            descriptor=ParameterDescriptor(
                description='Fraction of the survey area that must be scanned '
                'before the baseline is frozen (0.8 = 80%)'))
        self.declare_parameter('coverage_x_min', -30.0,
            descriptor=ParameterDescriptor(description='Survey area X lower bound (m)'))
        self.declare_parameter('coverage_x_max', 30.0,
            descriptor=ParameterDescriptor(description='Survey area X upper bound (m)'))
        self.declare_parameter('coverage_y_min', -30.0,
            descriptor=ParameterDescriptor(description='Survey area Y lower bound (m)'))
        self.declare_parameter('coverage_y_max', 30.0,
            descriptor=ParameterDescriptor(description='Survey area Y upper bound (m)'))
        self.declare_parameter('change_min_cluster_cells', 8,
            descriptor=ParameterDescriptor(
                description='Minimum connected cells for a change alert. Small '
                '3-6 cell patches at crown edges are noise; a real clearing is larger.'))
        self.declare_parameter('low_streak_threshold', 3,
            descriptor=ParameterDescriptor(
                description='Consecutive scans a cell must read as ground before it '
                            'counts as lost, so one stray beam cannot flip it.'))
        self.declare_parameter('change_min_baseline_hits', 3,
            descriptor=ParameterDescriptor(
                description='A cell can only be reported lost or gained if at least '
                            'this many scans saw it before the baseline.'))
        self.declare_parameter('baseline_canopy_fraction', 0.6,
            descriptor=ParameterDescriptor(
                description='A cell can only be lost if it read canopy in at least '
                            'this fraction of its baseline scans. Crown edges switch '
                            'between canopy and ground from beam to beam, so they are '
                            'not trusted.'))
        self.declare_parameter('baseline_ground_fraction', 0.2,
            descriptor=ParameterDescriptor(
                description='A cell can only be gained if it read canopy in at most '
                            'this fraction of its baseline scans (it was ground).'))
        self.declare_parameter('publish_rate', 1.0,
            descriptor=ParameterDescriptor(description='Map publish rate in Hz'))
        self.declare_parameter('max_cloud_points', 200000,
            descriptor=ParameterDescriptor(
                description='Maximum swath points retained in /drone_lidar_points'))

        self.res = self.get_parameter('resolution').value
        self.map_x = self.get_parameter('map_size_x').value
        self.map_y = self.get_parameter('map_size_y').value
        self.scan_topic = self.get_parameter('scan_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.map_frame = self.get_parameter('map_frame').value
        self.sensor_pitch = self.get_parameter('sensor_pitch').value
        self.sensor_yaw = self.get_parameter('sensor_yaw').value
        self.range_min = self.get_parameter('range_min').value
        self.alt = self.get_parameter('altitude').value
        self.canopy_threshold = self.get_parameter('canopy_threshold').value
        self.height_scale = self.get_parameter('height_scale').value
        self.baseline_threshold = self.get_parameter('baseline_threshold').value
        self.coverage_required = self.get_parameter('coverage_required').value
        self.coverage_x_min = self.get_parameter('coverage_x_min').value
        self.coverage_x_max = self.get_parameter('coverage_x_max').value
        self.coverage_y_min = self.get_parameter('coverage_y_min').value
        self.coverage_y_max = self.get_parameter('coverage_y_max').value
        self.change_min_cells = self.get_parameter('change_min_cluster_cells').value
        self.low_streak_threshold = self.get_parameter('low_streak_threshold').value
        self.change_min_baseline_hits = self.get_parameter('change_min_baseline_hits').value
        self.baseline_canopy_fraction = float(
            self.get_parameter('baseline_canopy_fraction').value)
        self.baseline_ground_fraction = float(
            self.get_parameter('baseline_ground_fraction').value)
        self.max_cloud_pts = self.get_parameter('max_cloud_points').value

        self.dim_x = int(self.map_x / self.res)
        self.dim_y = int(self.map_y / self.res)
        self.origin_x = -self.map_x / 2.0
        self.origin_y = -self.map_y / 2.0

        # Coverage is measured only inside the survey box. The map is
        # 80x80 m but the drone patrols a smaller box (60x60 m in dense).
        self.coverage_ix_min = max(0, int((self.coverage_x_min - self.origin_x) / self.res))
        self.coverage_ix_max = min(self.dim_x, int((self.coverage_x_max - self.origin_x) / self.res))
        self.coverage_iy_min = max(0, int((self.coverage_y_min - self.origin_y) / self.res))
        self.coverage_iy_max = min(self.dim_y, int((self.coverage_y_max - self.origin_y) / self.res))

        # Grids, indexed [ix, iy].
        # Highest height ever seen per cell (saved in the baseline).
        self.height_grid = np.zeros((self.dim_x, self.dim_y), dtype=np.float32)
        # Latest height per cell (drops when a tree is removed).
        self.height_recent = np.zeros((self.dim_x, self.dim_y), dtype=np.float32)
        # Consecutive scans each cell read as ground.
        self.low_streak = np.zeros((self.dim_x, self.dim_y), dtype=np.int16)
        # Consecutive scans each cell read as canopy.
        self.high_streak = np.zeros((self.dim_x, self.dim_y), dtype=np.int16)
        # Cumulative hit count per cell
        self.hits_grid = np.zeros((self.dim_x, self.dim_y), dtype=np.float32)
        # Per cell: how many scans saw it, and how many of those read canopy
        # (one count per scan, not per point).
        self.touch_count = np.zeros((self.dim_x, self.dim_y), dtype=np.int32)
        self.canopy_reads = np.zeros((self.dim_x, self.dim_y), dtype=np.int32)
        # Baseline snapshot
        self.baseline_heights = None
        # Cells scanned before the baseline. Cells first seen later are not
        # reported as new canopy.
        self.baseline_hits = None
        self.baseline_hit_counts = None     # scans that saw each cell
        self.baseline_canopy_frac = None    # fraction of those reading canopy
        self.baseline_established = False
        self.scan_count = 0

        # Change tracking stats
        self.total_canopy_lost = 0
        self.total_canopy_gained = 0
        self._alerted_cells = set()  # (ix//10, iy//10) blocks already alerted
        self._alerted_gained_cells = set()  # same, for gained clusters

        # Recent swath points (x, y, z) for RViz
        self.swath_pts = []

        # Drone pose from odometry
        self._odom_pos = np.zeros(3, dtype=np.float64)
        self._odom_q = Quaternion(w=1.0)
        self._odom_received = False

        # QoS profiles
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
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

        # Subscribers
        self.create_subscription(
            LaserScan, self.scan_topic, self.scan_callback, sensor_qos
        )
        self.create_subscription(
            Odometry, self.odom_topic, self.odom_callback, sensor_qos
        )

        # Publishers
        self.canopy_pub = self.create_publisher(
            OccupancyGrid, '/forest_canopy_map', map_qos
        )
        self.hits_pub = self.create_publisher(
            OccupancyGrid, '/forest_canopy_hits', map_qos
        )
        self.change_pub = self.create_publisher(
            OccupancyGrid, '/canopy_change_map', map_qos
        )
        self.change_events_pub = self.create_publisher(
            String, '/canopy_change_events', default_qos
        )
        self.change_marker_pub = self.create_publisher(
            MarkerArray, '/canopy_change_markers', default_qos
        )
        self.swath_pub = self.create_publisher(
            PointCloud2, '/drone_lidar_points', sensor_qos
        )
        self.baseline_pub = self.create_publisher(
            String, '/drone_baseline_status', default_qos
        )
        self.coverage_pub = self.create_publisher(
            String, '/scan_coverage', default_qos
        )

        # Reset baseline service
        self.reset_srv = self.create_service(
            Trigger, '~/reset_baseline', self.reset_baseline_callback
        )

        pub_period = 1.0 / max(self.get_parameter('publish_rate').value, 0.1)
        self.timer = self.create_timer(pub_period, self.publish_maps)

        self.get_logger().info(
            f'Canopy Scan Mapper ready. Grid: {self.dim_x}x{self.dim_y} '
            f'at {self.res}m, frame={self.map_frame}, altitude={self.alt}m. '
            f'Sensor pitch={math.degrees(self.sensor_pitch):.0f}°, '
            f'canopy threshold={self.canopy_threshold}m, '
            f'baseline after {self.baseline_threshold} scans.'
        )

    # Pose and projection helpers

    def odom_callback(self, msg: Odometry):
        """Store the drone pose from odometry."""
        self._odom_pos[0] = msg.pose.pose.position.x
        self._odom_pos[1] = msg.pose.pose.position.y
        self._odom_pos[2] = msg.pose.pose.position.z
        self._odom_q = msg.pose.pose.orientation
        self._odom_received = True

    def _coverage_cells(self):
        """Return (scanned_cells, total_cells) inside the survey area."""
        # The grid is indexed [ix, iy].
        sub = self.hits_grid[
            self.coverage_ix_min:self.coverage_ix_max,
            self.coverage_iy_min:self.coverage_iy_max,
        ]
        if sub.size == 0:
            return 0, 1
        return int(np.count_nonzero(sub > 0)), int(sub.size)

    # Scan processing

    def scan_callback(self, scan: LaserScan):
        """Project one scan line into the map frame and accumulate the grid."""
        if not self._odom_received:
            self.get_logger().warn(
                'No odometry received yet — skipping scan',
                throttle_duration_sec=10.0)
            return

        ranges = np.array(scan.ranges, dtype=np.float64)
        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
        valid = (np.isfinite(ranges)
                 & (ranges >= self.range_min)
                 & (ranges <= scan.range_max))
        if not np.any(valid):
            return

        r = ranges[valid]
        a = angles[valid]

        # Beam end points in the world frame. z uses the fixed altitude
        # parameter because the sim odometry always reports z=0.
        pts_world = project_scan_to_world(
            r, a, self._odom_q, self._odom_pos, self.alt,
            pitch=self.sensor_pitch, yaw=self.sensor_yaw)
        xs, ys, zs = pts_world[:, 0], pts_world[:, 1], pts_world[:, 2]

        # Keep recent swath points for RViz
        self.swath_pts.extend(pts_world.tolist())
        if len(self.swath_pts) > self.max_cloud_pts:
            self.swath_pts = self.swath_pts[-self.max_cloud_pts:]

        # Update the grids:
        #  - height_grid:   highest value ever seen (saved in the baseline)
        #  - height_recent: latest value. A removed tree's cells drop to
        #                   ground when they are scanned again.
        ix = ((xs - self.origin_x) / self.res).astype(np.int32)
        iy = ((ys - self.origin_y) / self.res).astype(np.int32)
        in_bounds = (ix >= 0) & (ix < self.dim_x) & (iy >= 0) & (iy < self.dim_y)
        ix, iy, zs = ix[in_bounds], iy[in_bounds], zs[in_bounds]
        if len(ix) == 0:
            return

        # Highest value ever seen.
        taller = zs > self.height_grid[ix, iy]
        self.height_grid[ix[taller], iy[taller]] = zs[taller]

        # Latest value. A cell is "touched" when any return of this scan
        # falls in it. Ground projects to about -0.2 m, so -inf marks
        # "no return" and heights are clamped at 0.
        scan_heights = np.full_like(self.height_recent, -np.inf)
        np.maximum.at(scan_heights, (ix, iy), zs)
        touched = np.isfinite(scan_heights)
        self.height_recent[touched] = np.maximum(scan_heights[touched], 0.0)

        # Only cells seen in this scan update their streaks. Cells not seen
        # keep their streak, so not being scanned never counts as a change.
        low_touched = self.height_recent[touched] < self.canopy_threshold
        self.touch_count[touched] += 1
        self.canopy_reads[touched] += ~low_touched
        self.low_streak[touched] = np.where(
            low_touched, self.low_streak[touched] + 1, 0)
        self.high_streak[touched] = np.where(
            ~low_touched, self.high_streak[touched] + 1, 0)

        np.add.at(self.hits_grid, (ix, iy), 1)

        self.scan_count += 1

        # Take the baseline once the required part of the survey area has
        # been scanned.
        covered, coverage_total = self._coverage_cells()
        coverage_pct = covered / coverage_total
        if (not self.baseline_established
                and self.scan_count >= self.baseline_threshold
                and coverage_pct >= self.coverage_required):
            self._snapshot_baseline()

        if self.scan_count % 50 == 0:
            covered_full = np.count_nonzero(self.hits_grid > 0)
            canopy_cells = np.count_nonzero(
                (self.hits_grid > 0) & (self.height_recent > self.canopy_threshold))
            status = 'BASELINE' if not self.baseline_established else 'MONITORING'
            loss_str = f' | Lost cells: {self.total_canopy_lost}' if self.baseline_established else ''
            self.get_logger().info(
                f'[{status}] Scan {self.scan_count}: {len(ix)} pts in grid, '
                f'{covered_full} cells covered '
                f'({coverage_pct * 100:.1f}% of survey area), '
                f'{canopy_cells} canopy cells{loss_str}'
            )

    # Baseline management

    def _snapshot_baseline(self):
        """Save the current grids as the baseline.

        Returns True if a snapshot was taken. Refuses until the required
        survey coverage is reached, so the baseline is never a partial map.
        """
        covered, coverage_total = self._coverage_cells()
        coverage_pct = covered / coverage_total
        if coverage_pct < self.coverage_required:
            self.get_logger().warn(
                f'Baseline blocked: {coverage_pct * 100:.1f}% scanned, '
                f'need {self.coverage_required * 100:.0f}%.',
                throttle_duration_sec=5.0,
            )
            return False

        self.baseline_heights = self.height_grid.copy()
        self.baseline_hits = self.hits_grid > 0
        self.baseline_hit_counts = self.touch_count.copy()
        self.baseline_canopy_frac = (
            self.canopy_reads / np.maximum(self.touch_count, 1)).astype(np.float32)
        self.baseline_established = True
        self._alerted_cells.clear()
        self._alerted_gained_cells.clear()
        canopy_cells = np.count_nonzero(self.baseline_heights > self.canopy_threshold)
        self.get_logger().info(
            f'BASELINE SNAPSHOT: {canopy_cells} canopy cells frozen '
            f'({covered} cells covered, {coverage_pct * 100:.1f}% of survey area). '
            f'Change detection active. '
            f'Call /scan_mapper/reset_baseline to re-baseline.'
        )
        self.baseline_pub.publish(String(
            data=f'baseline_established:{canopy_cells} '
                 f'coverage:{coverage_pct * 100:.1f}%'
        ))
        return True

    def reset_baseline_callback(self, request, response):
        """Service: reset the baseline to the current map."""
        if not self._snapshot_baseline():
            response.success = False
            response.message = (
                f'Coverage too low. Need {self.coverage_required * 100:.0f}% '
                'of the survey area scanned before re-baseline.'
            )
            return response
        self.total_canopy_lost = 0
        self.total_canopy_gained = 0
        response.success = True
        response.message = (
            f'Baseline reset. {np.count_nonzero(self.baseline_heights > self.canopy_threshold)} '
            f'canopy cells in new baseline.'
        )
        return response

    # Change detection

    def _compute_change(self):
        """Compare the current canopy with the baseline.

        Returns:
            change_grid: int8 grid (-100 = lost, 0 = unchanged, +100 = new)
            lost_mask: cells where canopy disappeared
            gained_mask: cells where canopy appeared
        """
        if self.baseline_heights is None:
            return None, None, None

        # Lost: the cell read canopy in most baseline scans and has read
        # ground for low_streak_threshold scans in a row. Gained: the reverse.
        current_canopy = self.height_recent > self.canopy_threshold
        well_seen = self.baseline_hit_counts >= self.change_min_baseline_hits
        frac = self.baseline_canopy_frac
        lost_mask = (well_seen & (frac >= self.baseline_canopy_fraction)
                     & ~current_canopy
                     & (self.low_streak >= self.low_streak_threshold))
        gained_mask = (well_seen & (frac <= self.baseline_ground_fraction)
                       & current_canopy
                       & (self.high_streak >= self.low_streak_threshold))

        change_grid = np.zeros((self.dim_x, self.dim_y), dtype=np.int8)
        change_grid[lost_mask] = -100
        change_grid[gained_mask] = 100

        return change_grid, lost_mask, gained_mask

    def _detect_change_clusters(self, binary_mask):
        """Find connected regions in a change mask.

        Returns a list of dicts: {evidence, cx, cy, area_cells, area_m2,
        world_x, world_y}.
        """
        if not np.any(binary_mask):
            return []
        labels, num_features = ndimage.label(binary_mask)
        clusters = []
        for label_id in range(1, num_features + 1):
            region = labels == label_id
            area = np.sum(region)
            if area < self.change_min_cells:
                continue
            # center_of_mass returns (ix, iy)
            ix_c, iy_c = ndimage.center_of_mass(region)
            # Cell centres are at (index + 0.5) * res.
            wx = self.origin_x + (ix_c + 0.5) * self.res
            wy = self.origin_y + (iy_c + 0.5) * self.res
            evidence = ''
            if self.baseline_hit_counts is not None:
                bh = self.baseline_hit_counts[region]
                b_h = self.baseline_heights[region]
                c_h = self.height_recent[region]
                bf = self.baseline_canopy_frac[region]
                evidence = (
                    f'baseline scans min/med={int(bh.min())}/{int(np.median(bh))}, '
                    f'baseline canopy frac med={float(np.median(bf)):.2f}, '
                    f'baseline h med={float(np.median(b_h)):.1f}m, '
                    f'now h med={float(np.median(c_h)):.1f}m')
            clusters.append({
                'evidence': evidence,
                'cx': int(ix_c), 'cy': int(iy_c),
                'area_cells': int(area),
                'area_m2': float(area * self.res**2),
                'world_x': float(wx),
                'world_y': float(wy),
            })
        return clusters

    def _alert_new_clusters(self, clusters, alerted, title, kind, log):
        """Alert for clusters not alerted before.

        A cluster is identified by the 10x10-cell block of its centre, kept
        in ``alerted``. Returns (alert strings, number of cells alerted).
        """
        alerts, cells = [], 0
        for cl in clusters:
            bucket = (cl['cx'] // 10, cl['cy'] // 10)
            if bucket in alerted:
                continue
            alerted.add(bucket)
            cells += cl['area_cells']
            alert = (
                f"{title}: ~{cl['area_cells']} cells ({cl['area_m2']:.0f}m²) "
                f"near ({cl['world_x']:.1f}, {cl['world_y']:.1f})"
            )
            self.get_logger().info(f'  {kind} evidence: {cl["evidence"]}')
            alerts.append(alert)
            log(alert)
        return alerts, cells

    def _cluster_markers(self, clusters, ns, rgb, first_id, stamp):
        """Cylinder markers for up to 50 clusters, ids from first_id."""
        markers = []
        for i, cl in enumerate(clusters[:50]):
            m = Marker()
            m.header = Header(stamp=stamp, frame_id=self.map_frame)
            m.id = first_id + i
            m.ns = ns
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose = Pose(
                position=Point(x=cl['world_x'], y=cl['world_y'], z=5.0),
                orientation=Quaternion(w=1.0),
            )
            m.scale.x = max(cl['area_m2'] ** 0.5, 1.5)
            m.scale.y = m.scale.x
            m.scale.z = 10.0
            m.color.r, m.color.g, m.color.b = rgb
            m.color.a = 0.9
            markers.append(m)
        return markers

    # Publishing

    def publish_maps(self):
        now = self.get_clock().now().to_msg()

        covered, coverage_total = self._coverage_cells()
        coverage_pct = covered / coverage_total
        self.coverage_pub.publish(String(
            data=f'coverage={coverage_pct * 100:.1f}% '
                 f'scanned={covered} total={coverage_total} '
                 f'required={self.coverage_required * 100:.0f}%'
        ))

        # 1. Canopy height map
        scanned = self.hits_grid > 0
        data = np.full((self.dim_x, self.dim_y), -1, dtype=np.int8)
        data[scanned] = np.clip(
            (self.height_recent[scanned] * self.height_scale), 0, 100
        ).astype(np.int8)

        grid = OccupancyGrid()
        grid.header = Header(stamp=now, frame_id=self.map_frame)
        grid.info.resolution = self.res
        grid.info.width = self.dim_x
        grid.info.height = self.dim_y
        grid.info.origin = Pose(
            position=Point(x=self.origin_x, y=self.origin_y, z=0.0),
            orientation=Quaternion(w=1.0)
        )
        # Grids are [ix, iy] but OccupancyGrid data is row-major [iy, ix],
        # so transpose.
        grid.data = data.T.flatten().tolist()
        self.canopy_pub.publish(grid)

        # Hit count per cell, so other nodes can ignore cells seen only once.
        hits_data = np.clip(self.hits_grid, 0, 100).astype(np.int8)
        hits_grid_msg = OccupancyGrid()
        hits_grid_msg.header = Header(stamp=now, frame_id=self.map_frame)
        hits_grid_msg.info = grid.info
        hits_grid_msg.data = hits_data.T.flatten().tolist()
        self.hits_pub.publish(hits_grid_msg)

        # 2. Change map (compared with the baseline)
        change_grid, lost_mask, gained_mask = self._compute_change()

        if change_grid is not None:
            change_msg = OccupancyGrid()
            change_msg.header = Header(stamp=now, frame_id=self.map_frame)
            change_msg.info = grid.info
            change_msg.data = change_grid.T.flatten().tolist()
            self.change_pub.publish(change_msg)

            # 3. Change alerts for large enough regions
            lost_clusters = self._detect_change_clusters(lost_mask)
            gained_clusters = self._detect_change_clusters(gained_mask)

            lost_alerts, lost_cells = self._alert_new_clusters(
                lost_clusters, self._alerted_cells, 'CANOPY LOST', 'lost',
                self.get_logger().warn)
            gained_alerts, gained_cells = self._alert_new_clusters(
                gained_clusters, self._alerted_gained_cells, 'NEW CANOPY',
                'gained', self.get_logger().info)
            self.total_canopy_lost += lost_cells
            self.total_canopy_gained += gained_cells
            new_alerts = lost_alerts + gained_alerts
            if new_alerts:
                self.change_events_pub.publish(String(data='; '.join(new_alerts)))

            # 4. Change markers (red = lost, blue = gained)
            markers = MarkerArray()
            markers.markers += self._cluster_markers(
                lost_clusters, 'canopy_lost', (1.0, 0.0, 0.0), 0, now)
            markers.markers += self._cluster_markers(
                gained_clusters, 'canopy_gained', (0.0, 0.5, 1.0),
                len(markers.markers), now)
            if markers.markers:
                self.change_marker_pub.publish(markers)

        # 5. Swath point cloud
        if self.swath_pts:
            pts = np.array(self.swath_pts, dtype=np.float32)
            cloud = PointCloud2()
            cloud.header = Header(stamp=now, frame_id=self.map_frame)
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
            cloud.data = pts.tobytes()
            self.swath_pub.publish(cloud)

    def destroy_node(self):
        self.get_logger().info(
            f'Canopy Scan Mapper shutting down. {self.scan_count} scans, '
            f'{len(self.swath_pts)} swath points. '
            f'Lost cells: {self.total_canopy_lost}, Gained: {self.total_canopy_gained}.'
        )
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ScanMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
