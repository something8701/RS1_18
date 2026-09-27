#!/usr/bin/env python3
"""
Camera republisher: turns raw sensor_msgs/Image into a JPEG CompressedImage
feed for the dashboard.

image_transport's `republish` ignores the `out` remap in ROS 2 Humble (it
always publishes /out/compressed), so this small node sets its topic names
itself. Run one per robot; the in/out topics are parameters.
"""

import time

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv2


class CameraRepublisher(Node):
    """Republishes images as JPEG, rate-limited."""

    def __init__(self):
        super().__init__('camera_republisher')

        self.declare_parameter('in_topic', '/parrot1/camera/image',
            descriptor=ParameterDescriptor(
                description='Raw Image topic to subscribe to'))
        self.declare_parameter('out_topic', '/parrot1/camera/compressed',
            descriptor=ParameterDescriptor(
                description='CompressedImage topic to publish on'))
        self.declare_parameter('jpeg_quality', 70,
            descriptor=ParameterDescriptor(
                description='JPEG encode quality (1-100)'))
        self.declare_parameter('max_rate', 5.0,
            descriptor=ParameterDescriptor(
                description='Max publish rate in Hz (wall clock, independent '
                            'of sim time)'))

        in_topic = self.get_parameter('in_topic').value
        out_topic = self.get_parameter('out_topic').value
        self.quality = int(self.get_parameter('jpeg_quality').value)
        self.min_interval = 1.0 / max(0.1, float(self.get_parameter('max_rate').value))

        # The Gazebo camera bridge publishes best effort.
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        # The dashboard subscribes through rosbridge with reliable QoS.
        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self._bridge = CvBridge()
        self._last_pub = 0.0
        self._frames = 0
        self.pub = self.create_publisher(CompressedImage, out_topic, pub_qos)
        self.create_subscription(Image, in_topic, self._on_image, sensor_qos)

        self.get_logger().info(
            f'Republishing {in_topic} -> {out_topic} '
            f'(JPEG q={self.quality}, max {1.0/self.min_interval:.1f} Hz)'
        )

    def _on_image(self, msg: Image):
        """Encode the latest frame as JPEG and publish it, rate-limited."""
        now = time.time()
        if now - self._last_pub < self.min_interval:
            return
        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as exc:
            self.get_logger().warn(
                f'CV bridge error: {exc}', throttle_duration_sec=30.0)
            return
        ok, buf = cv2.imencode(
            '.jpg', cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        if not ok:
            self.get_logger().warn(
                'JPEG encode failed', throttle_duration_sec=30.0)
            return
        out = CompressedImage()
        out.header = msg.header
        out.format = 'jpeg'
        out.data = buf.tobytes()
        self.pub.publish(out)
        self._last_pub = now
        self._frames += 1

    def destroy_node(self):
        self.get_logger().info(
            f'Camera republisher shutting down. {self._frames} frames sent.')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraRepublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
