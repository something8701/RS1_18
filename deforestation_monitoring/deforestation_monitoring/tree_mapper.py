#!/usr/bin/env python3
"""
Tree Trunk Mapper: Uses the Husky's 2D lidar to detect and map tree trunks.

Principle: Tree trunks appear as tight circular clusters of lidar returns
at ground level. As the Husky patrols, these clusters persist at fixed
world locations. We cluster lidar points, track persistent clusters, and
build a tree map.

Once a baseline is established, every subsequent publish compares the
current tree map against the baseline and publishes:
  - /forest_change_map  — OccupancyGrid: -100=lost, 0=unchanged, +100=new
  - /forest_change_events — String alerts for significant cluster losses
"""

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import numpy as np
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point, Pose, Quaternion
from std_msgs.msg import Header, String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformException
from tf2_ros.transform_listener import TransformListener
from sklearn.cluster import DBSCAN
from scipy import ndimage


class TreeMapper(Node):
    """Maps tree trunk locations from Husky 2D lidar, detects changes from baseline."""

    def __init__(self):
        super().__init__('tree_mapper')

        # -- Parameters --
        self.declare_parameter('resolution', 0.5,
            descriptor=ParameterDescriptor(description='Grid cell size in metres'))
        self.declare_parameter('map_size_x', 80.0,
            descriptor=ParameterDescriptor(description='Map X dimension in metres'))
        self.declare_parameter('map_size_y', 80.0,
            descriptor=ParameterDescriptor(description='Map Y dimension in metres'))
        self.declare_parameter('publish_rate', 1.0,
            descriptor=ParameterDescriptor(description='Map publish rate in Hz'))
        self.declare_parameter('cluster_eps', 0.6,
            descriptor=ParameterDescriptor(description='DBSCAN epsilon (m)'))
        self.declare_parameter('min_cluster_size', 2,
            descriptor=ParameterDescriptor(description='Minimum points to form a tree trunk'))
        self.declare_parameter('trunk_radius', 0.25,
            descriptor=ParameterDescriptor(description='Typical tree trunk radius (m)'))
        self.declare_parameter('baseline_threshold', 100,
            descriptor=ParameterDescriptor(description='Scans before baseline is established'))
        self.declare_parameter('tree_confidence_threshold', 2,
            descriptor=ParameterDescriptor(
                description='Minimum lidar hits for a cell to count as a confirmed tree'))
        self.declare_parameter('change_min_cluster_cells', 3,
            descriptor=ParameterDescriptor(
                description='Minimum connected cells to trigger a change alert'))
        self.declare_parameter('hit_decay', 0.9,
            descriptor=ParameterDescriptor(
                description='Per-scan decay factor for the recent-tree hit count. '
                            'A value < 1 lets removed trunks fall below the '
                            'confidence threshold so they can be reported as lost.'))

        self.res = self.get_parameter('resolution').value
        self.map_x = self.get_parameter('map_size_x').value
        self.map_y = self.get_parameter('map_size_y').value
        self.cluster_eps = self.get_parameter('cluster_eps').value
        self.min_cluster = self.get_parameter('min_cluster_size').value
        self.baseline_threshold = self.get_parameter('baseline_threshold').value
        self.tree_confidence = self.get_parameter('tree_confidence_threshold').value
        self.change_min_cells = self.get_parameter('change_min_cluster_cells').value
        self.hit_decay = self.get_parameter('hit_decay').value

        self.dim_x = int(self.map_x / self.res)
        self.dim_y = int(self.map_y / self.res)
        self.origin_x = -self.map_x / 2.0
        self.origin_y = -self.map_y / 2.0

        # --- Grids ---
        # Cumulative hit count per cell (all-time — used for the trunk map and
        # the baseline snapshot, which must include every tree ever seen).
        self.tree_hits = np.zeros((self.dim_x, self.dim_y), dtype=np.float32)
        # Decaying recent hit count (current state — decays each processed scan
        # so a removed trunk eventually falls below the confidence threshold).
        self.recent_hits = np.zeros((self.dim_x, self.dim_y), dtype=np.float32)
        # Cells covered by the lidar since the last baseline. Only re-observed
        # cells are eligible for "lost" detection, so areas never revisited
        # after the baseline are not falsely reported as cleared.
        self.swept = np.zeros((self.dim_x, self.dim_y), dtype=np.float32)
        # Baseline snapshot: frozen copy taken when baseline is established
        self.baseline_hits = None
        # Whether the baseline snapshot has been taken
        self.baseline_established = False
        self.scan_count = 0

        # Change tracking stats
        self.total_trees_lost = 0
        self.total_trees_gained = 0
        self.last_change_alert = ''
        self._alerted_cells = set()  # (ix, iy) of already-alerted loss cells
        self._alerted_gained_cells = set()  # same, for gained clusters

        # DBSCAN clusterer
        self.clusterer = DBSCAN(eps=self.cluster_eps, min_samples=self.min_cluster)

        # TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

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
        self.scan_sub = self.create_subscription(
            LaserScan, '/husky1/scan', self.scan_callback, sensor_qos
        )

        # Publishers
        # Trunk map (Husky lidar). /forest_canopy_map is owned by the drone's
        # scan_mapper — two grids on one topic would corrupt pattern_scanner.
        self.trunk_pub = self.create_publisher(
            OccupancyGrid, '/forest_trunk_map', map_qos
        )
        self.change_pub = self.create_publisher(
            OccupancyGrid, '/forest_change_map', map_qos
        )
        self.change_events_pub = self.create_publisher(
            String, '/forest_change_events', default_qos
        )
        self.marker_pub = self.create_publisher(
            MarkerArray, '/tree_markers', default_qos
        )
        self.change_marker_pub = self.create_publisher(
            MarkerArray, '/tree_change_markers', default_qos
        )
        self.cluster_points_pub = self.create_publisher(
            PointCloud2, '/tree_cluster_points', sensor_qos
        )
        self.all_tree_points_pub = self.create_publisher(
            PointCloud2, '/all_tree_positions', default_qos
        )
        self.baseline_pub = self.create_publisher(
            String, '/baseline_status', default_qos
        )

        # Reset baseline service
        self.reset_srv = self.create_service(
            Trigger, '~/reset_baseline', self.reset_baseline_callback
        )

        pub_period = 1.0 / max(self.get_parameter('publish_rate').value, 0.1)
        self.timer = self.create_timer(pub_period, self.publish_maps)

        self.get_logger().info(
            f'Tree Mapper ready. Grid: {self.dim_x}x{self.dim_y} '
            f'at {self.res}m. Tree confidence: >= {self.tree_confidence} hits. '
            f'Change alert min cluster: {self.change_min_cells} cells.'
        )

    # ── Lidar processing ──────────────────────────────────────────────

    def _scan_to_xy(self, scan: LaserScan):
        """Convert LaserScan ranges to (x,y) points in base_link frame."""
        angles = scan.angle_min + np.arange(len(scan.ranges)) * scan.angle_increment
        ranges = np.array(scan.ranges)
        valid = np.isfinite(ranges) & (ranges > scan.range_min) & (ranges < scan.range_max)
        angles = angles[valid]
        ranges = ranges[valid]
        if len(ranges) < self.min_cluster:
            return np.zeros((0, 2))
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        return np.column_stack([x, y])

    def _to_world(self, pts_local):
        """Transform points from base_link to husky1_map frame."""
        try:
            tf = self.tf_buffer.lookup_transform(
                'husky1_map', 'husky1_base_link', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
        except TransformException:
            return None

        t = tf.transform.translation
        q = tf.transform.rotation
        xx, yy, zz = q.x*q.x, q.y*q.y, q.z*q.z
        xy = q.x * q.y
        wz = q.w * q.z

        R = np.array([
            [1-2*(yy+zz), 2*(xy-wz)],
            [2*(xy+wz),   1-2*(xx+zz)],
        ], dtype=np.float32)

        pts_w = pts_local @ R.T
        pts_w[:, 0] += t.x
        pts_w[:, 1] += t.y
        return pts_w

    def scan_callback(self, scan: LaserScan):
        points = self._scan_to_xy(scan)
        if len(points) < self.min_cluster:
            return

        pts_world = self._to_world(points)
        if pts_world is None or len(pts_world) < self.min_cluster:
            return

        # Decay the recent hit count every processed scan, then re-add detections
        # below. This is what makes change detection able to report "lost": a
        # cumulative count can only grow, so a removed tree would stay "present".
        self.recent_hits *= self.hit_decay

        # Decay the recent sweep mask in lockstep with recent_hits, then mark
        # every lidar return's cell as swept. A removed trunk only counts as
        # lost while its location is being actively re-covered; once the robot
        # drives away both recent_hits AND swept decay together, so a still-
        # present tree that is simply out of view is never reported lost.
        self.swept *= self.hit_decay
        six = ((pts_world[:, 0] - self.origin_x) / self.res).astype(np.int32)
        siy = ((pts_world[:, 1] - self.origin_y) / self.res).astype(np.int32)
        s_in = (six >= 0) & (six < self.dim_x) & (siy >= 0) & (siy < self.dim_y)
        self.swept[six[s_in], siy[s_in]] = 1.0

        # DBSCAN clustering
        labels = self.clusterer.fit_predict(pts_world)

        unique_labels = set(labels)
        tree_count = 0
        cluster_centroids = []
        for label in unique_labels:
            if label == -1:
                continue
            cluster_pts = pts_world[labels == label]
            if len(cluster_pts) < self.min_cluster:
                continue
            cx, cy = np.mean(cluster_pts, axis=0)
            cluster_centroids.append([cx, cy, 0.0])
            ix = int((cx - self.origin_x) / self.res)
            iy = int((cy - self.origin_y) / self.res)
            if 0 <= ix < self.dim_x and 0 <= iy < self.dim_y:
                self.tree_hits[ix, iy] += 1.0
                self.recent_hits[ix, iy] += 1.0
                tree_count += 1

        # Publish live cluster points
        if cluster_centroids:
            pts = np.array(cluster_centroids, dtype=np.float32)
            cloud = PointCloud2()
            cloud.header = Header(stamp=scan.header.stamp, frame_id='husky1_map')
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
            self.cluster_points_pub.publish(cloud)

        self.scan_count += 1

        # Baseline: snapshot the current hit map
        if not self.baseline_established and self.scan_count >= self.baseline_threshold:
            self._snapshot_baseline()

        if self.scan_count % 50 == 0:
            total_trees = np.count_nonzero(self.tree_hits >= self.tree_confidence)
            status = 'BASELINE' if not self.baseline_established else 'MONITORING'
            loss_str = f' | Lost: {self.total_trees_lost}' if self.baseline_established else ''
            self.get_logger().info(
                f'[{status}] Scan {self.scan_count}: {tree_count} trunks this scan, '
                f'{total_trees} confirmed tree cells{loss_str}'
            )

    # ── Baseline management ───────────────────────────────────────────

    def _snapshot_baseline(self):
        """Freeze the current tree_hits as the baseline for change detection."""
        self.baseline_hits = self.tree_hits.copy()
        self.baseline_established = True
        # Only lidar coverage AFTER this baseline counts as re-observation.
        self.swept[:] = 0.0
        self._alerted_cells.clear()
        self._alerted_gained_cells.clear()
        total = np.count_nonzero(self.baseline_hits >= self.tree_confidence)
        self.get_logger().info(
            f'BASELINE SNAPSHOT: {total} confirmed tree cells frozen. '
            f'Change detection active. Call /tree_mapper/reset_baseline to re-baseline.'
        )
        self.baseline_pub.publish(String(data=f'baseline_established:{total}'))

    def reset_baseline_callback(self, request, response):
        """Service callback: manually reset the baseline to current state."""
        self._snapshot_baseline()
        self.total_trees_lost = 0
        self.total_trees_gained = 0
        self.last_change_alert = ''
        response.success = True
        response.message = (
            f'Baseline reset. {np.count_nonzero(self.baseline_hits >= self.tree_confidence)} '
            f'trees in new baseline.'
        )
        return response

    # ── Change detection ──────────────────────────────────────────────

    def _compute_change(self):
        """Compare current tree_hits against baseline.

        Returns:
            change_grid: np.int8 array (-100 = lost, 0 = unchanged, +100 = new)
            lost_mask: boolean array of cells where trees were lost
            gained_mask: boolean array of cells where trees appeared
        """
        if self.baseline_hits is None:
            return None, None, None

        # Binarize: a cell is currently a tree if its RECENT hit count is high
        # enough. The baseline uses the cumulative count (every tree seen).
        current_trees = self.recent_hits >= self.tree_confidence
        baseline_trees = self.baseline_hits >= self.tree_confidence

        # A tree is lost only if it was in the baseline, its location has been
        # re-observed since baseline, and it is no longer detected now.
        lost_mask = baseline_trees & (self.swept >= 0.5) & ~current_trees
        gained_mask = current_trees & ~baseline_trees

        change_grid = np.zeros((self.dim_x, self.dim_y), dtype=np.int8)
        change_grid[lost_mask] = -100
        change_grid[gained_mask] = 100

        return change_grid, lost_mask, gained_mask

    def _detect_change_clusters(self, binary_mask, min_cells=None):
        """Run connected-component labelling on a binary change mask.

        Returns list of dicts: {cx, cy, area_cells, area_m2, world_x, world_y}
        """
        if min_cells is None:
            min_cells = self.change_min_cells
        if not np.any(binary_mask):
            return []
        labels, num_features = ndimage.label(binary_mask)
        clusters = []
        for label_id in range(1, num_features + 1):
            region = labels == label_id
            area = np.sum(region)
            if area < min_cells:
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

        # --- 1. Tree trunk map (current state) ---
        if np.max(self.tree_hits) > 0:
            max_hit = max(1.0, np.max(self.tree_hits))
            data = np.clip((self.tree_hits / max_hit) * 100, 0, 100).astype(np.int8)
        else:
            data = np.full(self.dim_x * self.dim_y, -1, dtype=np.int8)

        grid = OccupancyGrid()
        grid.header = Header(stamp=now, frame_id='husky1_map')
        grid.info.resolution = self.res
        grid.info.width = self.dim_x
        grid.info.height = self.dim_y
        grid.info.origin = Pose(
            position=Point(x=self.origin_x, y=self.origin_y, z=0.0),
            orientation=Quaternion(w=1.0)
        )
        # Grids are indexed [ix, iy]; OccupancyGrid data is row-major [iy, ix].
        grid.data = data.T.flatten().tolist()
        self.trunk_pub.publish(grid)

        # --- 2. Change map (diff vs baseline) ---
        change_grid, lost_mask, gained_mask = self._compute_change()

        if change_grid is not None:
            change_msg = OccupancyGrid()
            change_msg.header = Header(stamp=now, frame_id='husky1_map')
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
                self.total_trees_lost += cl['area_cells']
                alert = (
                    f"TREES LOST: ~{cl['area_cells']} trees ({cl['area_m2']:.0f}m²) "
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
                self.total_trees_gained += cl['area_cells']
                alert = (
                    f"NEW GROWTH: ~{cl['area_cells']} trees ({cl['area_m2']:.0f}m²) "
                    f"near ({cl['world_x']:.1f}, {cl['world_y']:.1f})"
                )
                new_alerts.append(alert)
                self.get_logger().info(alert)

            # Publish all new alerts as a single semicolon-delimited message
            if new_alerts:
                self.last_change_alert = '; '.join(new_alerts)
                self.change_events_pub.publish(String(data=self.last_change_alert))

            # --- 4. Change markers (red = lost, blue = gained) ---
            change_markers = MarkerArray()
            marker_id = 0
            for cl in lost_clusters[:50]:
                m = Marker()
                m.header = Header(stamp=now, frame_id='husky1_map')
                m.id = marker_id; marker_id += 1
                m.ns = 'tree_lost'
                m.type = Marker.CYLINDER
                m.action = Marker.ADD
                m.pose = Pose(
                    position=Point(x=cl['world_x'], y=cl['world_y'], z=2.5),
                    orientation=Quaternion(w=1.0),
                )
                m.scale.x = max(cl['area_m2'] ** 0.5, 1.5)
                m.scale.y = m.scale.x
                m.scale.z = 5.0
                m.color.r = 1.0; m.color.g = 0.0; m.color.b = 0.0; m.color.a = 0.9
                change_markers.markers.append(m)

            for cl in gained_clusters[:50]:
                m = Marker()
                m.header = Header(stamp=now, frame_id='husky1_map')
                m.id = marker_id; marker_id += 1
                m.ns = 'tree_gained'
                m.type = Marker.CYLINDER
                m.action = Marker.ADD
                m.pose = Pose(
                    position=Point(x=cl['world_x'], y=cl['world_y'], z=2.5),
                    orientation=Quaternion(w=1.0),
                )
                m.scale.x = max(cl['area_m2'] ** 0.5, 1.5)
                m.scale.y = m.scale.x
                m.scale.z = 5.0
                m.color.r = 0.0; m.color.g = 0.5; m.color.b = 1.0; m.color.a = 0.9
                change_markers.markers.append(m)

            if change_markers.markers:
                self.change_marker_pub.publish(change_markers)

        # --- 5. Tree markers (current) ---
        markers = MarkerArray()
        tree_cells = np.argwhere(self.tree_hits >= self.tree_confidence)
        for i, (ix, iy) in enumerate(tree_cells[:200]):
            wx = self.origin_x + (ix + 0.5) * self.res
            wy = self.origin_y + (iy + 0.5) * self.res
            m = Marker()
            m.header = Header(stamp=now, frame_id='husky1_map')
            m.id = i
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose = Pose(position=Point(x=wx, y=wy, z=0.0),
                           orientation=Quaternion(w=1.0))
            m.scale.x = 0.3
            m.scale.y = 0.3
            m.scale.z = 5.0
            m.color.r = 0.0
            m.color.g = 1.0
            m.color.b = 0.0
            m.color.a = 0.8
            markers.markers.append(m)
        self.marker_pub.publish(markers)

        # --- 6. All tree positions as PointCloud2 ---
        if len(tree_cells) > 0:
            pts_all = np.column_stack([
                self.origin_x + (tree_cells[:, 0] + 0.5) * self.res,
                self.origin_y + (tree_cells[:, 1] + 0.5) * self.res,
                np.zeros(len(tree_cells)),
            ]).astype(np.float32)
            cloud = PointCloud2()
            cloud.header = Header(stamp=now, frame_id='husky1_map')
            cloud.height = 1
            cloud.width = len(pts_all)
            cloud.fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            ]
            cloud.is_bigendian = False
            cloud.point_step = 12
            cloud.row_step = cloud.point_step * cloud.width
            cloud.is_dense = True
            cloud.data = pts_all.tobytes()
            self.all_tree_points_pub.publish(cloud)

    def destroy_node(self):
        self.get_logger().info(
            f'Tree Mapper shutting down. '
            f'Lost: {self.total_trees_lost}, Gained: {self.total_trees_gained}.'
        )
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TreeMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
