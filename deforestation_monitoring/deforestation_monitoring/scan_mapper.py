#!/usr/bin/env python3
"""
Canopy Scan Mapper: Builds a 2.5D canopy height map from the Parrot
drone's downward-pitched 2D LiDAR.

The LiDAR (360 samples, 0-2π) is pitched ~90° down on the drone, so each
scan is a push-broom slice through the forest below. Each beam is rotated
by the lidar's static mounting pose (pitch/yaw params) and the drone's
full odometry orientation into the map frame, then accumulated into a
max-height-per-cell grid.

Outputs:
  - /forest_canopy_map    OccupancyGrid: canopy height 0-100 (0-10 m),
                          -1 = not yet scanned
  - /drone_lidar_points   PointCloud2: accumulated swath points (verification)
  - /canopy_change_map    OccupancyGrid: -100 = canopy lost, +100 = new
  - /canopy_change_events String alerts for connected lost regions
  - /canopy_change_markers MarkerArray (red = lost, blue = gained)
  - /drone_baseline_status String

Change detection compares against a baseline snapshot taken after N scans.
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
from collections import deque
import math


class ScanMapper(Node):
    """Maps canopy height from the drone's pitched LiDAR swaths."""

    def __init__(self):
        super().__init__('scan_mapper')

        # -- Parameters --
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
                description='Drone flight altitude (m). The sim odometry reports z=0 '
                'regardless of the real pose, so the projection uses this fixed altitude.'))
        self.declare_parameter('canopy_threshold', 2.0,
            descriptor=ParameterDescriptor(
                description='Cell height (m) above which a cell counts as canopy'))
        self.declare_parameter('height_scale', 10.0,
            descriptor=ParameterDescriptor(
                description='Map value = min(height * height_scale, 100); 10 → 0-10 m'))
        self.declare_parameter('baseline_threshold', 100,
            descriptor=ParameterDescriptor(
                description='Minimum scans before the baseline snapshot may trigger'))
        self.declare_parameter('baseline_loop_radius', 8.0,
            descriptor=ParameterDescriptor(
                description='Loop-closure radius (m): the snapshot fires when the drone returns '
                'within this distance of terrain it surveyed long ago — i.e. the first '
                'lawnmower pass is complete.'))
        self.declare_parameter('baseline_loop_scans', 300,
            descriptor=ParameterDescriptor(
                description='Minimum scan separation for a position revisit to count as loop '
                'closure (a revisit of recent terrain is just normal flight)'))
        self.declare_parameter('change_min_cluster_cells', 8,
            descriptor=ParameterDescriptor(
                description='Minimum connected cells to trigger a change alert. '
                'Raised above the default 3 because patrol path drift keeps discovering '
                '3-6 cell edge speckles after baseline — a real clearing is far larger.'))
        self.declare_parameter('height_drop_threshold', 3.0,
            descriptor=ParameterDescriptor(
                description='Minimum height drop (m) between baseline and current for a cell '
                            'to count as lost. Hysteresis kills edge flicker.'))
        self.declare_parameter('low_streak_threshold', 2,
            descriptor=ParameterDescriptor(
                description='Consecutive scans a cell must read as not-canopy before it '
                            'counts as lost. A single grazing beam cannot flip a cell.'))
        self.declare_parameter('enable_change_detection', True,
            descriptor=ParameterDescriptor(
                description='Run the built-in baseline/change detection. Set false when '
                            'the standalone change_detector node owns /canopy_change_map, '
                            '/canopy_change_events, /canopy_change_markers and '
                            '/drone_baseline_status.'))
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
        self.loop_radius = self.get_parameter('baseline_loop_radius').value
        self.loop_scans = self.get_parameter('baseline_loop_scans').value
        self.change_min_cells = self.get_parameter('change_min_cluster_cells').value
        self.height_drop = self.get_parameter('height_drop_threshold').value
        self.low_streak_threshold = self.get_parameter('low_streak_threshold').value
        self.max_cloud_pts = self.get_parameter('max_cloud_points').value
        self.change_enabled = self.get_parameter('enable_change_detection').value

        self.dim_x = int(self.map_x / self.res)
        self.dim_y = int(self.map_y / self.res)
        self.origin_x = -self.map_x / 2.0
        self.origin_y = -self.map_y / 2.0

        # --- Grids ---
        # All-time running max height per cell (used for the baseline snapshot)
        self.height_grid = np.zeros((self.dim_x, self.dim_y), dtype=np.float32)
        # Windowed max height (current state — can fall when a tree is removed)
        self.height_recent = np.zeros((self.dim_x, self.dim_y), dtype=np.float32)
        # Consecutive scans where each cell read as "not canopy" (hysteresis
        # so a single grazing beam cannot flip a cell to "lost")
        self.low_streak = np.zeros((self.dim_x, self.dim_y), dtype=np.int16)
        # Consecutive scans where each cell read as "canopy" (hysteresis for
        # "gained" so a single canopy graze cannot flip a cell to "new")
        self.high_streak = np.zeros((self.dim_x, self.dim_y), dtype=np.int16)
        # Cumulative hit count per cell
        self.hits_grid = np.zeros((self.dim_x, self.dim_y), dtype=np.float32)
        # Baseline snapshot: frozen copy taken when baseline is established
        self.baseline_heights = None
        # Cells that had been scanned at baseline time — gates "new canopy"
        # so cells first surveyed after baseline are not falsely reported.
        self.baseline_hits = None
        self.baseline_established = False
        self.scan_count = 0
        # (scan_count, x, y) history for loop-closure baseline triggering
        self._pos_history = deque(maxlen=3000)

        # Change tracking stats
        self.total_canopy_lost = 0
        self.total_canopy_gained = 0
        self._alerted_cells = set()  # (ix//10, iy//10) of already-alerted loss cells
        self._alerted_gained_cells = set()  # same, for gained clusters

        # Accumulated swath points (x, y, z) for RViz verification
        self.swath_pts = []

        # Drone pose from odometry
        self._odom_pos = np.zeros(3, dtype=np.float64)
        self._odom_q = Quaternion(w=1.0)
        self._odom_received = False

        # -- QoS profiles --
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
        if self.change_enabled:
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
        if self.change_enabled:
            self.baseline_pub = self.create_publisher(
                String, '/drone_baseline_status', default_qos
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

    # ── Pose and projection helpers ──────────────────────────────────

    def odom_callback(self, msg: Odometry):
        """Track the drone's full 3D pose from odometry."""
        self._odom_pos[0] = msg.pose.pose.position.x
        self._odom_pos[1] = msg.pose.pose.position.y
        self._odom_pos[2] = msg.pose.pose.position.z
        self._odom_q = msg.pose.pose.orientation
        self._odom_received = True

    def _rotation_matrix(self, q: Quaternion) -> np.ndarray:
        """Full 3x3 rotation matrix from a quaternion."""
        x, y, z, w = q.x, q.y, q.z, q.w
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        return np.array([
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ], dtype=np.float64)

    def _beam_directions_body(self, angles: np.ndarray) -> np.ndarray:
        """Beam unit vectors in the drone body frame.

        The 2D lidar's native beams lie in its XY plane: (cos a, sin a, 0).
        The sensor is mounted with a static pitch about Y and yaw about Z
        (applied pitch first, then yaw), giving the push-broom orientation.
        """
        n = len(angles)
        d = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(n)])

        cp, sp = math.cos(self.sensor_pitch), math.sin(self.sensor_pitch)
        R_pitch = np.array([
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ])
        cy, sy = math.cos(self.sensor_yaw), math.sin(self.sensor_yaw)
        R_yaw = np.array([
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ])
        # body = R_yaw @ R_pitch @ sensor
        return d @ R_pitch.T @ R_yaw.T

    # ── Scan processing ───────────────────────────────────────────────

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

        d_body = self._beam_directions_body(a)
        R = self._rotation_matrix(self._odom_q)

        # World-frame endpoints: p_w = R_body_world @ (r * d_body) + pos.
        # z comes from the fixed altitude param — the sim odometry always
        # reports z=0 even though the drone flies at the patrol altitude.
        pos = self._odom_pos.copy()
        pos[2] = self.alt
        pts_world = (d_body * r[:, None]) @ R.T + pos
        xs, ys, zs = pts_world[:, 0], pts_world[:, 1], pts_world[:, 2]

        # Accumulate swath points for visualization
        self.swath_pts.extend(pts_world.tolist())
        if len(self.swath_pts) > self.max_cloud_pts:
            self.swath_pts = self.swath_pts[-self.max_cloud_pts:]

        # Accumulate into the grids:
        #  - height_grid   : all-time running max (frozen into the baseline)
        #  - height_recent : per-scan max overwrite (current state). A removed
        #                    tree's cell falls to ground on re-survey, while a
        #                    single grazing beam cannot erase a canopy cell
        #                    because the next canopy scan restores it.
        ix = ((xs - self.origin_x) / self.res).astype(np.int32)
        iy = ((ys - self.origin_y) / self.res).astype(np.int32)
        in_bounds = (ix >= 0) & (ix < self.dim_x) & (iy >= 0) & (iy < self.dim_y)
        ix, iy, zs = ix[in_bounds], iy[in_bounds], zs[in_bounds]
        if len(ix) == 0:
            return

        # All-time running max (baseline).
        taller = zs > self.height_grid[ix, iy]
        self.height_grid[ix[taller], iy[taller]] = zs[taller]

        # Per-scan max overwrite (current state).
        scan_heights = np.full_like(self.height_recent, -1.0)
        np.maximum.at(scan_heights, (ix, iy), zs)
        touched = scan_heights >= 0.0
        self.height_recent[touched] = scan_heights[touched]

        # Hysteresis: only OBSERVED cells advance their streak. A cell must be
        # observed "not canopy" for >= low_streak_threshold consecutive scans
        # before it counts as lost; unobserved cells keep their last streak so
        # the counter cannot be satisfied by simply not being seen again.
        low_touched = self.height_recent[touched] < self.canopy_threshold
        self.low_streak[touched] = np.where(
            low_touched, self.low_streak[touched] + 1, 0)
        self.high_streak[touched] = np.where(
            ~low_touched, self.high_streak[touched] + 1, 0)

        np.add.at(self.hits_grid, (ix, iy), 1)

        self.scan_count += 1

        # Baseline triggers on loop closure: the drone returns to terrain it
        # surveyed long ago, meaning the first lawnmower pass is complete.
        # This avoids freezing the baseline mid-survey, when every newly
        # seen tree would read as "NEW CANOPY" in the change map.
        covered = np.count_nonzero(self.hits_grid > 0)
        self._pos_history.append(
            (self.scan_count, float(self._odom_pos[0]), float(self._odom_pos[1])))
        if (self.change_enabled and not self.baseline_established
                and self.scan_count >= self.baseline_threshold):
            for sc, ax, ay in self._pos_history:
                if self.scan_count - sc < self.loop_scans:
                    break  # history is chronological — rest is too recent
                if math.hypot(self._odom_pos[0] - ax, self._odom_pos[1] - ay) < self.loop_radius:
                    self._snapshot_baseline()
                    break

        if self.scan_count % 50 == 0:
            canopy_cells = np.count_nonzero(
                (self.hits_grid > 0) & (self.height_recent > self.canopy_threshold))
            status = 'BASELINE' if not self.baseline_established else 'MONITORING'
            loss_str = f' | Lost cells: {self.total_canopy_lost}' if self.baseline_established else ''
            self.get_logger().info(
                f'[{status}] Scan {self.scan_count}: {len(ix)} pts in grid, '
                f'{covered} cells covered, {canopy_cells} canopy cells{loss_str}'
            )

    # ── Baseline management ───────────────────────────────────────────

    def _snapshot_baseline(self):
        """Freeze the current height grid as the baseline."""
        self.baseline_heights = self.height_grid.copy()
        self.baseline_hits = self.hits_grid > 0
        self.baseline_established = True
        self._alerted_cells.clear()
        self._alerted_gained_cells.clear()
        canopy_cells = np.count_nonzero(self.baseline_heights > self.canopy_threshold)
        covered = np.count_nonzero(self.hits_grid > 0)
        self.get_logger().info(
            f'BASELINE SNAPSHOT: {canopy_cells} canopy cells frozen '
            f'({covered} cells covered). Change detection active. '
            f'Call /scan_mapper/reset_baseline to re-baseline.'
        )
        self.baseline_pub.publish(String(data=f'baseline_established:{canopy_cells}'))

    def reset_baseline_callback(self, request, response):
        """Service callback: manually reset the baseline to current state."""
        self._snapshot_baseline()
        self.total_canopy_lost = 0
        self.total_canopy_gained = 0
        response.success = True
        response.message = (
            f'Baseline reset. {np.count_nonzero(self.baseline_heights > self.canopy_threshold)} '
            f'canopy cells in new baseline.'
        )
        return response

    # ── Change detection ──────────────────────────────────────────────

    def _compute_change(self):
        """Compare current canopy against baseline.

        Returns:
            change_grid: np.int8 array (-100 = lost, 0 = unchanged, +100 = new)
            lost_mask: boolean array of cells where canopy disappeared
            gained_mask: boolean array of cells where canopy appeared
        """
        if self.baseline_heights is None:
            return None, None, None

        # Height-drop detection with hysteresis. A cell is lost only if its
        # height fell by at least height_drop AND it is now below the canopy
        # threshold — a small 5.0 -> 4.8 m flicker at a canopy edge does not
        # count. "New" is the symmetric case (rose by height_drop).
        baseline_h = self.baseline_heights
        current_h = self.height_recent
        baseline_canopy = baseline_h > self.canopy_threshold
        current_canopy = current_h > self.canopy_threshold

        lost_mask = (baseline_canopy & (baseline_h - current_h > self.height_drop)
                     & ~current_canopy
                     & (self.low_streak >= self.low_streak_threshold))
        gained_mask = (current_canopy & (current_h - baseline_h > self.height_drop)
                       & ~baseline_canopy & self.baseline_hits
                       & (self.high_streak >= self.low_streak_threshold))

        change_grid = np.zeros((self.dim_x, self.dim_y), dtype=np.int8)
        change_grid[lost_mask] = -100
        change_grid[gained_mask] = 100

        return change_grid, lost_mask, gained_mask

    def _detect_change_clusters(self, binary_mask):
        """Connected-component labelling on a binary change mask.

        Returns list of dicts: {cx, cy, area_cells, area_m2, world_x, world_y}
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
            # center_of_mass returns (axis-0, axis-1) = (ix, iy) centres
            ix_c, iy_c = ndimage.center_of_mass(region)
            wx = self.origin_x + ix_c * self.res
            wy = self.origin_y + iy_c * self.res
            clusters.append({
                'cx': int(ix_c), 'cy': int(iy_c),
                'area_cells': int(area),
                'area_m2': float(area * self.res**2),
                'world_x': float(wx),
                'world_y': float(wy),
            })
        return clusters

    # ── Publishing ────────────────────────────────────────────────────

    def publish_maps(self):
        now = self.get_clock().now().to_msg()

        # --- 1. Canopy height map (current state) ---
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
        # Grids are indexed [ix, iy]; OccupancyGrid data is row-major [iy, ix]
        # so the transpose is required (dim_x == dim_y hides it, but it still
        # swaps x/y for any non-square pattern).
        grid.data = data.T.flatten().tolist()
        self.canopy_pub.publish(grid)

        # --- 2. Change map (diff vs baseline) ---
        change_grid, lost_mask, gained_mask = self._compute_change()

        if change_grid is not None and self.change_enabled:
            change_msg = OccupancyGrid()
            change_msg.header = Header(stamp=now, frame_id=self.map_frame)
            change_msg.info = grid.info
            change_msg.data = change_grid.T.flatten().tolist()
            self.change_pub.publish(change_msg)

            # --- 3. Change events (alerts for significant clusters) ---
            lost_clusters = self._detect_change_clusters(lost_mask)
            gained_clusters = self._detect_change_clusters(gained_mask)

            new_alerts = []
            for cl in lost_clusters:
                # Dedup: skip if we already alerted near this cell
                bucket = (cl['cx'] // 10, cl['cy'] // 10)
                if bucket in self._alerted_cells:
                    continue
                self._alerted_cells.add(bucket)
                self.total_canopy_lost += cl['area_cells']
                alert = (
                    f"CANOPY LOST: ~{cl['area_cells']} cells ({cl['area_m2']:.0f}m²) "
                    f"near ({cl['world_x']:.1f}, {cl['world_y']:.1f})"
                )
                new_alerts.append(alert)
                self.get_logger().warn(alert)

            for cl in gained_clusters:
                # Dedup: skip if we already alerted near this cell
                bucket = (cl['cx'] // 10, cl['cy'] // 10)
                if bucket in self._alerted_gained_cells:
                    continue
                self._alerted_gained_cells.add(bucket)
                self.total_canopy_gained += cl['area_cells']
                alert = (
                    f"NEW CANOPY: ~{cl['area_cells']} cells ({cl['area_m2']:.0f}m²) "
                    f"near ({cl['world_x']:.1f}, {cl['world_y']:.1f})"
                )
                new_alerts.append(alert)
                self.get_logger().info(alert)

            if new_alerts:
                self.change_events_pub.publish(String(data='; '.join(new_alerts)))

            # --- 4. Change markers (red = lost, blue = gained) ---
            markers = MarkerArray()
            marker_id = 0
            for cl in lost_clusters[:50]:
                m = Marker()
                m.header = Header(stamp=now, frame_id=self.map_frame)
                m.id = marker_id; marker_id += 1
                m.ns = 'canopy_lost'
                m.type = Marker.CYLINDER
                m.action = Marker.ADD
                m.pose = Pose(
                    position=Point(x=cl['world_x'], y=cl['world_y'], z=5.0),
                    orientation=Quaternion(w=1.0),
                )
                m.scale.x = max(cl['area_m2'] ** 0.5, 1.5)
                m.scale.y = m.scale.x
                m.scale.z = 10.0
                m.color.r = 1.0; m.color.g = 0.0; m.color.b = 0.0; m.color.a = 0.9
                markers.markers.append(m)

            for cl in gained_clusters[:50]:
                m = Marker()
                m.header = Header(stamp=now, frame_id=self.map_frame)
                m.id = marker_id; marker_id += 1
                m.ns = 'canopy_gained'
                m.type = Marker.CYLINDER
                m.action = Marker.ADD
                m.pose = Pose(
                    position=Point(x=cl['world_x'], y=cl['world_y'], z=5.0),
                    orientation=Quaternion(w=1.0),
                )
                m.scale.x = max(cl['area_m2'] ** 0.5, 1.5)
                m.scale.y = m.scale.x
                m.scale.z = 10.0
                m.color.r = 0.0; m.color.g = 0.5; m.color.b = 1.0; m.color.a = 0.9
                markers.markers.append(m)

            if markers.markers:
                self.change_marker_pub.publish(markers)

        # --- 5. Swath point cloud ---
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
