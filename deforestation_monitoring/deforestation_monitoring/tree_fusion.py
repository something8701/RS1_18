#!/usr/bin/env python3
"""
Tree fusion: combines tree detections from two sensors.

Inputs:
  - /all_tree_positions (PointCloud2): Husky 2D lidar trunk clusters (DBSCAN)
  - /drone_terrain (PointCloud2): drone camera canopy points (green pixels)

Outputs:
  - /fused_tree_map (OccupancyGrid): 0 = no tree, 33 = lidar only,
    66 = camera only, 100 = both
  - /fusion_markers (MarkerArray): green = both, yellow = lidar only,
    blue = camera only

Points from each sensor are counted per grid cell, and a cell counts for a
sensor once it has enough points:
  - LIDAR_ONLY (33): trunk seen by the lidar only (for example under canopy)
  - CAMERA_ONLY (66): canopy seen by the camera only (for example a hidden trunk)
  - BOTH (100): seen by both sensors, the most reliable
"""

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import numpy as np
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point, Pose, Quaternion
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray


# Occupancy values for the fused map
LIDAR_ONLY = 33
CAMERA_ONLY = 66
BOTH = 100
NO_TREE = 0


class TreeFusion(Node):
    """Combines lidar and camera tree detections into one map."""

    def __init__(self):
        super().__init__('tree_fusion')

        # Parameters
        self.declare_parameter('resolution', 0.5,
            descriptor=ParameterDescriptor(description='Grid resolution (m)'))
        self.declare_parameter('map_size_x', 80.0,
            descriptor=ParameterDescriptor(description='Map X size (m)'))
        self.declare_parameter('map_size_y', 80.0,
            descriptor=ParameterDescriptor(description='Map Y size (m)'))
        self.declare_parameter('publish_rate', 1.0,
            descriptor=ParameterDescriptor(description='Map publish rate (Hz)'))
        self.declare_parameter('lidar_min_points', 2,
            descriptor=ParameterDescriptor(
                description='Minimum lidar points in a cell to count as a lidar tree'))
        self.declare_parameter('camera_min_points', 2,
            descriptor=ParameterDescriptor(
                description='Minimum camera points in a cell to count as a camera tree'))
        self.declare_parameter('fusion_frame', 'husky1_map',
            descriptor=ParameterDescriptor(description='Output frame for fused map'))

        self.res = self.get_parameter('resolution').value
        self.map_x = self.get_parameter('map_size_x').value
        self.map_y = self.get_parameter('map_size_y').value
        self.lidar_min = self.get_parameter('lidar_min_points').value
        self.camera_min = self.get_parameter('camera_min_points').value
        self.fusion_frame = self.get_parameter('fusion_frame').value

        self.dim_x = int(self.map_x / self.res)
        self.dim_y = int(self.map_y / self.res)
        self.origin_x = -self.map_x / 2.0
        self.origin_y = -self.map_y / 2.0

        # Accumulator grids
        self.lidar_grid = np.zeros((self.dim_x, self.dim_y), dtype=np.int32)
        self.camera_grid = np.zeros((self.dim_x, self.dim_y), dtype=np.int32)
        self.lidar_points_received = 0
        self.camera_points_received = 0

        # QoS
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
        self.lidar_sub = self.create_subscription(
            PointCloud2, '/all_tree_positions', self.lidar_cb, sensor_qos)
        self.camera_sub = self.create_subscription(
            PointCloud2, '/drone_terrain', self.camera_cb, sensor_qos)

        # Publishers
        self.fused_pub = self.create_publisher(
            OccupancyGrid, '/fused_tree_map', map_qos)
        self.fusion_marker_pub = self.create_publisher(
            MarkerArray, '/fusion_markers', default_qos)

        rate = max(self.get_parameter('publish_rate').value, 0.5)
        self.create_timer(1.0 / rate, self.publish_fusion)

        self.get_logger().info(
            f'Tree Fusion ready. Grid: {self.dim_x}x{self.dim_y} at {self.res}m. '
            f'Lidar min={self.lidar_min}, Camera min={self.camera_min}. '
            f'Subscribing to /all_tree_positions + /drone_terrain.'
        )

    # Point cloud ingestion

    def _cloud_to_cells(self, cloud: PointCloud2):
        """Return the (ix, iy) cells of the points in a PointCloud2.

        Expects x, y, z as FLOAT32 at offsets 0, 4, 8.
        """
        if cloud is None or cloud.width == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

        # Raw bytes to an (x, y, z) array
        pts = np.frombuffer(cloud.data, dtype=np.float32).reshape(-1, cloud.point_step // 4)
        xs = pts[:, 0]
        ys = pts[:, 1]

        ix = ((xs - self.origin_x) / self.res).astype(np.int32)
        iy = ((ys - self.origin_y) / self.res).astype(np.int32)

        valid = (ix >= 0) & (ix < self.dim_x) & (iy >= 0) & (iy < self.dim_y)
        return ix[valid], iy[valid]

    def lidar_cb(self, cloud: PointCloud2):
        """Count lidar tree detections."""
        ix, iy = self._cloud_to_cells(cloud)
        if len(ix) == 0:
            return
        np.add.at(self.lidar_grid, (ix, iy), 1)
        self.lidar_points_received += len(ix)

    def camera_cb(self, cloud: PointCloud2):
        """Count camera canopy detections.

        The drone_terrain cloud has both tree and ground points, so only
        points above 2 m are counted.
        """
        if cloud is None or cloud.width == 0:
            return

        pts = np.frombuffer(cloud.data, dtype=np.float32).reshape(-1, cloud.point_step // 4)
        xs = pts[:, 0]
        ys = pts[:, 1]
        zs = pts[:, 2]

        # Only canopy height points (above 2 m), not ground
        canopy_mask = zs > 2.0
        xs = xs[canopy_mask]
        ys = ys[canopy_mask]

        if len(xs) == 0:
            return

        ix = ((xs - self.origin_x) / self.res).astype(np.int32)
        iy = ((ys - self.origin_y) / self.res).astype(np.int32)
        valid = (ix >= 0) & (ix < self.dim_x) & (iy >= 0) & (iy < self.dim_y)

        np.add.at(self.camera_grid, (ix[valid], iy[valid]), 1)
        self.camera_points_received += len(ix[valid])

    # Fusion publishing

    def publish_fusion(self):
        """Build the fused map and publish it with markers."""
        now = self.get_clock().now().to_msg()

        # Which cells count for each sensor
        lidar_trees = self.lidar_grid >= self.lidar_min
        camera_trees = self.camera_grid >= self.camera_min

        both_mask = lidar_trees & camera_trees
        lidar_only_mask = lidar_trees & ~camera_trees
        camera_only_mask = camera_trees & ~lidar_trees

        # Fused grid
        fused = np.full((self.dim_x, self.dim_y), NO_TREE, dtype=np.int8)
        fused[lidar_only_mask] = LIDAR_ONLY
        fused[camera_only_mask] = CAMERA_ONLY
        fused[both_mask] = BOTH

        # OccupancyGrid
        grid = OccupancyGrid()
        grid.header = Header(stamp=now, frame_id=self.fusion_frame)
        grid.info.resolution = self.res
        grid.info.width = self.dim_x
        grid.info.height = self.dim_y
        grid.info.origin = Pose(
            position=Point(x=self.origin_x, y=self.origin_y, z=0.0),
            orientation=Quaternion(w=1.0),
        )
        # Grids are [ix, iy] but OccupancyGrid data is row-major [iy, ix].
        grid.data = fused.T.flatten().tolist()
        self.fused_pub.publish(grid)

        # Markers
        markers = MarkerArray()
        mid = 0

        # Green: both sensors
        for ix, iy in np.argwhere(both_mask)[:100]:
            markers.markers.append(self._make_marker(
                mid, ix, iy, now, 'both', 0.0, 1.0, 0.0))
            mid += 1

        # Yellow: lidar only
        for ix, iy in np.argwhere(lidar_only_mask)[:100]:
            markers.markers.append(self._make_marker(
                mid, ix, iy, now, 'lidar_only', 1.0, 1.0, 0.0))
            mid += 1

        # Blue: camera only
        for ix, iy in np.argwhere(camera_only_mask)[:100]:
            markers.markers.append(self._make_marker(
                mid, ix, iy, now, 'camera_only', 0.0, 0.5, 1.0))
            mid += 1

        if markers.markers:
            self.fusion_marker_pub.publish(markers)

        # Stats
        l_total = np.count_nonzero(lidar_trees)
        c_total = np.count_nonzero(camera_trees)
        b_total = np.count_nonzero(both_mask)
        lo_total = np.count_nonzero(lidar_only_mask)
        co_total = np.count_nonzero(camera_only_mask)

        self.get_logger().info(
            f'Fusion: {b_total} both | {lo_total} lidar-only | {co_total} camera-only | '
            f'({l_total} lidar total, {c_total} camera total)'
        )

    def _make_marker(self, mid, ix, iy, stamp, ns, r, g, b):
        """Create a cylinder marker at cell (ix, iy)."""
        wx = self.origin_x + (ix + 0.5) * self.res
        wy = self.origin_y + (iy + 0.5) * self.res
        m = Marker()
        m.header = Header(stamp=stamp, frame_id=self.fusion_frame)
        m.id = mid
        m.ns = ns
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.pose = Pose(position=Point(x=wx, y=wy, z=2.5),
                       orientation=Quaternion(w=1.0))
        m.scale.x = 0.4
        m.scale.y = 0.4
        m.scale.z = 5.0
        m.color.r = r
        m.color.g = g
        m.color.b = b
        m.color.a = 0.8
        return m

    def destroy_node(self):
        self.get_logger().info(
            f'Tree Fusion shutting down. '
            f'Lidar pts: {self.lidar_points_received}, '
            f'Camera pts: {self.camera_points_received}.'
        )
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TreeFusion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
