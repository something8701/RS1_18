#!/usr/bin/env python3
# depth_obstacles.py: The Husky's depth camera, thinned for Nav2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField


def thin_cloud(msg: PointCloud2, stride: int, max_range: float) -> np.ndarray:
    a = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.width, msg.point_step)
    if msg.height > 1:
        a = a[::stride, ::stride]
    else:
        a = a[:, ::stride * stride]
    xyz = a[..., :12].reshape(-1, 12).copy().view(np.float32).reshape(-1, 3)
    ok = np.isfinite(xyz).all(1)
    ok &= np.einsum('ij,ij->i', xyz, xyz) <= max_range * max_range
    return xyz[ok]


class DepthObstacles(Node):

    def __init__(self):
        super().__init__('depth_obstacles')
        p = lambda n, d: self.declare_parameter(n, d).value
        self.stride = int(p('stride', 4))
        self.max_range = float(p('max_range', 5.0))
        self.pub = self.create_publisher(
            PointCloud2, p('out_topic', '/husky1/camera/depth/obstacle_points'), 5)
        self.create_subscription(PointCloud2, p('in_topic', '/husky1/camera/depth/points'),
                                 self.cloud_cb, qos_profile_sensor_data)
        self.get_logger().info(
            f'Depth obstacles ready: every {self.stride}th row/column, up to {self.max_range} m')

    def cloud_cb(self, msg: PointCloud2):
        pts = thin_cloud(msg, self.stride, self.max_range)
        out = PointCloud2(header=msg.header, height=1, width=len(pts), is_dense=True,
                          is_bigendian=False, point_step=12, row_step=12 * len(pts),
                          fields=[PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32,
                                             count=1) for i, n in enumerate('xyz')])
        out.data = pts.astype(np.float32).tobytes()
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = DepthObstacles()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
