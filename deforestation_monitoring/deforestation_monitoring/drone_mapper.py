#!/usr/bin/env python3
"""
Drone Terrain Mapper.

Accumulates the Parrot's native depth point cloud, transforms it from the
camera frame into parrot1_odom, downsamples it, and republishes it as
`/drone_terrain` for the dashboard's 3D Terrain view.

The source point cloud is:

    /parrot1/camera/depth/points  (sensor_msgs/PointCloud2)

The dashboard already subscribes to `/drone_terrain`, so this node is the
single place where the raw depth cloud is turned into a world-frame terrain
cloud.
"""

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.time import Time
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformListener
from tf2_ros import ConnectivityException, ExtrapolationException, LookupException
import math
import numpy as np


class DroneMapper(Node):
    """Transform and accumulate the Parrot depth cloud into parrot1_odom."""

    def __init__(self):
        super().__init__('drone_mapper')

        self.declare_parameter('depth_topic', '/parrot1/camera/depth/points',
            descriptor=ParameterDescriptor(description='Parrot depth PointCloud2 topic'))
        self.declare_parameter('target_frame', 'parrot1_odom',
            descriptor=ParameterDescriptor(description='Frame to transform terrain into'))
        self.declare_parameter('altitude', 10.0,
            descriptor=ParameterDescriptor(description='Drone survey altitude above ground (m)'))
        self.declare_parameter('publish_rate', 2.0,
            descriptor=ParameterDescriptor(description='Terrain point cloud publish rate (Hz)'))
        self.declare_parameter('terrain_resolution', 0.25,
            descriptor=ParameterDescriptor(description='Terrain accumulation cell size (m)'))
        self.declare_parameter('max_publish_points', 50000,
            descriptor=ParameterDescriptor(description='Maximum points published per cloud'))
        self.declare_parameter('downsample_step', 12,
            descriptor=ParameterDescriptor(description='Keep every Nth valid depth point'))
        self.declare_parameter('process_every_n_frames', 1,
            descriptor=ParameterDescriptor(description='Process every Nth incoming cloud'))

        self.depth_topic = self.get_parameter('depth_topic').value
        self.target_frame = self.get_parameter('target_frame').value
        self.alt = float(self.get_parameter('altitude').value)
        self.res = max(0.05, float(self.get_parameter('terrain_resolution').value))
        self.max_publish_points = int(self.get_parameter('max_publish_points').value)
        self.step = max(1, int(self.get_parameter('downsample_step').value))
        self.process_every = max(1, int(self.get_parameter('process_every_n_frames').value))

        # cell key -> [count, sum_x, sum_y, sum_z]
        self.terrain_cells = {}
        self.frames = 0

        # Gazebo bridge sensor topics are best-effort in this project.
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        terrain_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cloud_sub = self.create_subscription(
            PointCloud2, self.depth_topic, self.cloud_cb, sensor_qos
        )
        self.terrain_pub = self.create_publisher(
            PointCloud2, '/drone_terrain', terrain_qos
        )
        # Lightweight copy for the dashboard: <=8000 points at <=1 Hz so a
        # single 3D view never saturates the rosbridge websocket.
        self.viz_pub = self.create_publisher(
            PointCloud2, '/drone_terrain_viz', terrain_qos
        )
        self._viz_tick = 0

        rate = max(0.5, float(self.get_parameter('publish_rate').value))
        self.create_timer(1.0 / rate, self.publish_terrain)

        self.get_logger().info(
            f'Drone Mapper ready. depth={self.depth_topic}, '
            f'target={self.target_frame}, altitude={self.alt}m, step={self.step}, '
            f'cell={self.res}m, max_publish={self.max_publish_points}'
        )

    def _lookup_transform(self, cloud: PointCloud2):
        """Return camera->target transform for this cloud, or None."""
        try:
            return self.tf_buffer.lookup_transform(
                self.target_frame,
                cloud.header.frame_id,
                Time(),
                Duration(seconds=0.5),
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as exc:
            self.get_logger().warn(
                f'TF lookup failed for {cloud.header.frame_id} -> '
                f'{self.target_frame}: {exc}',
                throttle_duration_sec=5.0,
            )
            return None

    def _extract_points(self, cloud: PointCloud2, transform):
        """Extract x/y/z points and transform them into the target frame."""
        offsets = {}
        for field in cloud.fields:
            if field.name in ('x', 'y', 'z'):
                offsets[field.name] = field.offset
        if len(offsets) != 3:
            self.get_logger().warn(
                'Depth cloud does not contain x/y/z fields',
                throttle_duration_sec=5.0,
            )
            return []

        point_step = cloud.point_step
        if point_step <= 0 or not cloud.data:
            return []

        raw = np.frombuffer(bytes(cloud.data), dtype=np.uint8)
        usable = raw.size - (raw.size % point_step)
        if usable <= 0:
            return []

        arr = raw[:usable].reshape(-1, point_step)
        float_dtype = '>f4' if cloud.is_bigendian else '<f4'

        def col(name):
            off = offsets[name]
            return arr[:, off:off + 4].copy().view(float_dtype).ravel()

        x = col('x')
        y = col('y')
        z = col('z')

        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        x = x[valid]
        y = y[valid]
        z = z[valid]

        x, y, z = self._rotate_and_translate(x, y, z, transform)
        z = z + self.alt
        valid = (np.abs(x) < 50.0) & (np.abs(y) < 50.0) & (z > -0.5) & (z < 40.0)
        x = x[valid]
        y = y[valid]
        z = z[valid]
        x = x[::self.step]
        y = y[::self.step]
        z = z[::self.step]

        return [
            (float(px), float(py), float(pz))
            for px, py, pz in zip(x, y, z)
        ]

    def _rotate_and_translate(self, x, y, z, transform):
        """Apply a TransformStamped to numpy point arrays."""
        tx = transform.transform.translation.x
        ty = transform.transform.translation.y
        tz = transform.transform.translation.z
        qx = transform.transform.rotation.x
        qy = transform.transform.rotation.y
        qz = transform.transform.rotation.z
        qw = transform.transform.rotation.w

        # v' = v + qw*t + q_vec x t, where t = 2*(q_vec x v)
        tw = 2.0 * (qy * z - qz * y)
        tyv = 2.0 * (qz * x - qx * z)
        tzv = 2.0 * (qx * y - qy * x)

        xr = x + qw * tw + (qy * tzv - qz * tyv)
        yr = y + qw * tyv + (qz * tw - qx * tzv)
        zr = z + qw * tzv + (qx * tyv - qy * tw)

        return xr + tx, yr + ty, zr + tz

    def cloud_cb(self, cloud: PointCloud2):
        """Transform and accumulate the latest depth cloud."""
        self.frames += 1
        if self.frames % self.process_every != 0:
            return

        transform = self._lookup_transform(cloud)
        if transform is None:
            return

        points = self._extract_points(cloud, transform)
        if not points:
            return

        for px, py, pz in points:
            key = (
                int(math.floor(px / self.res)),
                int(math.floor(py / self.res)),
            )
            cell = self.terrain_cells.get(key)
            if cell is None:
                self.terrain_cells[key] = [1, px, py, pz]
            else:
                cell[0] += 1
                cell[1] += px
                cell[2] += py
                cell[3] += pz

    def publish_terrain(self):
        """Publish the accumulated world-frame terrain as PointCloud2."""
        if not self.terrain_cells:
            return

        keys = sorted(self.terrain_cells.keys())
        if len(keys) > self.max_publish_points:
            indices = np.unique(
                np.linspace(0, len(keys) - 1, self.max_publish_points, dtype=np.int64)
            )
            keys = [keys[i] for i in indices]

        pts = np.array(
            [
                [
                    self.terrain_cells[key][1] / self.terrain_cells[key][0],
                    self.terrain_cells[key][2] / self.terrain_cells[key][0],
                    self.terrain_cells[key][3] / self.terrain_cells[key][0],
                ]
                for key in keys
            ],
            dtype=np.float32,
        )
        cloud = PointCloud2()
        cloud.header = Header(
            stamp=self.get_clock().now().to_msg(),
            frame_id=self.target_frame,
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
        cloud.data = pts.tobytes()
        self.terrain_pub.publish(cloud)

        # Dashboard terrain view (throttled + downsampled server-side).
        self._viz_tick += 1
        if self._viz_tick % 2 == 0:
            max_viz = 8000
            step = max(1, len(pts) // max_viz)
            viz_pts = pts[::step][:max_viz]
            viz = PointCloud2()
            viz.header = cloud.header
            viz.height = 1
            viz.width = len(viz_pts)
            viz.fields = cloud.fields
            viz.is_bigendian = False
            viz.point_step = 12
            viz.row_step = viz.point_step * viz.width
            viz.is_dense = True
            viz.data = viz_pts.tobytes()
            self.viz_pub.publish(viz)

        self.get_logger().info(
            f'Terrain: {len(self.terrain_cells)} cells, publishing {len(pts)} pts '
            f'from {self.depth_topic} in {self.target_frame}'
        )

    def destroy_node(self):
        self.get_logger().info(
            f'Drone Mapper shutting down. {len(self.terrain_cells)} cells collected.'
        )
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
