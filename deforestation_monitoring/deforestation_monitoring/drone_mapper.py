#!/usr/bin/env python3
"""
Drone Survey Mapper: Publishes the drone's camera ground footprint as a
3D point cloud. As the drone patrols, the terrain fills in with surveyed
points — green where canopy detected, brown for bare ground.

Uses the drone's RGB camera + odometry + heading to map pixel locations
to world coordinates.
"""

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import numpy as np
from sensor_msgs.msg import Image, PointCloud2, PointField
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
import cv2
import math


class DroneMapper(Node):
    """Maps terrain by projecting the drone's downward camera into world coords."""

    def __init__(self):
        super().__init__('drone_mapper')

        self.declare_parameter('publish_rate', 2.0,
            descriptor=ParameterDescriptor(description='Terrain point cloud publish rate (Hz)'))
        self.declare_parameter('altitude', 10.0,
            descriptor=ParameterDescriptor(description='Drone flight altitude (m)'))
        self.declare_parameter('fov_width', 15.0,
            descriptor=ParameterDescriptor(description='Ground footprint width at altitude (m)'))
        self.declare_parameter('sample_grid_x', 8,
            descriptor=ParameterDescriptor(description='Camera sample grid columns'))
        self.declare_parameter('sample_grid_y', 6,
            descriptor=ParameterDescriptor(description='Camera sample grid rows'))
        self.declare_parameter('max_points', 30000,
            descriptor=ParameterDescriptor(description='Maximum terrain points to retain'))

        self.alt = self.get_parameter('altitude').value
        self.fov_w = self.get_parameter('fov_width').value
        self.sample_x = self.get_parameter('sample_grid_x').value
        self.sample_y = self.get_parameter('sample_grid_y').value
        self.max_pts = self.get_parameter('max_points').value

        # Accumulated terrain points: (x, y, z, is_tree)
        self.terrain_pts = []

        self.bridge = CvBridge()
        self.drone_x = 2.0
        self.drone_y = 0.0
        self.drone_z = self.alt
        self.drone_yaw = 0.0  # heading in radians — now tracked from odometry
        self.frames = 0

        # -- QoS profiles --
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        default_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Subscribers
        self.create_subscription(Image, '/parrot1/camera/image', self.image_cb, sensor_qos)
        self.create_subscription(Odometry, '/parrot1/odometry', self.odom_cb, sensor_qos)

        # Publisher
        self.terrain_pub = self.create_publisher(PointCloud2, '/drone_terrain', default_qos)

        # Timer
        rate = max(self.get_parameter('publish_rate').value, 0.5)
        self.create_timer(1.0 / rate, self.publish_terrain)

        self.get_logger().info(
            f'Drone Mapper ready. Grid: {self.sample_x}x{self.sample_y}, '
            f'fov={self.fov_w}m, altitude={self.alt}m'
        )

    def odom_cb(self, msg: Odometry):
        """Track drone position and heading from odometry."""
        self.drone_x = msg.pose.pose.position.x
        self.drone_y = msg.pose.pose.position.y
        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.drone_yaw = math.atan2(siny, cosy)

    def image_cb(self, msg: Image):
        """Process camera frame: detect green (canopy) vs bare ground."""
        self.frames += 1
        if self.frames % 5 != 0:  # process every 5th frame
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError:
            self.get_logger().warn('cv_bridge conversion failed', throttle_duration_sec=5.0)
            return

        h, w = cv_image.shape[:2]
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, (25, 25, 20), (95, 255, 220))

        # Pre-compute rotation for this frame
        cos_h = math.cos(self.drone_yaw)
        sin_h = math.sin(self.drone_yaw)

        # Sample grid across the image footprint
        for sy in range(self.sample_y):
            for sx in range(self.sample_x):
                ix = int(w * (sx + 0.5) / self.sample_x)
                iy = int(h * (sy + 0.5) / self.sample_y)
                is_tree = bool(green_mask[iy, ix] > 0)

                # Map pixel to world offset in drone body frame
                px = (float(sx) / self.sample_x - 0.5) * self.fov_w
                py = (0.5 - float(sy) / self.sample_y) * self.fov_w * 0.7

                # Rotate by drone heading to get world-frame coordinates
                wx = self.drone_x + px * cos_h - py * sin_h
                wy = self.drone_y + px * sin_h + py * cos_h
                # Height: canopy (~4m) vs ground (~0.1m)
                wz = 4.0 if is_tree else 0.1

                self.terrain_pts.append((wx, wy, wz, is_tree))

        # Trim if too many points
        if len(self.terrain_pts) > self.max_pts:
            self.terrain_pts = self.terrain_pts[-self.max_pts:]

    def publish_terrain(self):
        """Publish accumulated terrain points as PointCloud2."""
        if not self.terrain_pts:
            return

        pts = np.array(self.terrain_pts, dtype=np.float32)
        cloud = PointCloud2()
        cloud.header = Header(
            stamp=self.get_clock().now().to_msg(),
            frame_id='parrot1_odom',
        )
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
        cloud.data = pts[:, :3].tobytes()
        self.terrain_pub.publish(cloud)

        canopy_count = sum(1 for p in self.terrain_pts if p[3])
        pct = canopy_count / max(len(self.terrain_pts), 1) * 100
        self.get_logger().info(
            f'Terrain: {len(self.terrain_pts)} pts, {pct:.0f}% canopy, '
            f'drone at ({self.drone_x:.1f}, {self.drone_y:.1f}), '
            f'heading={math.degrees(self.drone_yaw):.0f}°'
        )

    def destroy_node(self):
        self.get_logger().info(f'Drone Mapper shutting down. {len(self.terrain_pts)} points collected.')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DroneMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
