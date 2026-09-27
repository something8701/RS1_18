#!/usr/bin/env python3
"""Watch the simulation /clock and warn when it stops.

Gazebo Fortress sometimes stops publishing /clock (the whole simulation
hangs). If no clock message arrives for `stall_timeout` wall-clock seconds,
this node logs a warning and publishes a /sim_stalled String, so the time
of the stall is recorded.
"""

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String


class ClockWatchdog(Node):
    def __init__(self):
        super().__init__('clock_watchdog')
        self.declare_parameter('stall_timeout', 10.0)
        self.stall_timeout = self.get_parameter('stall_timeout').value
        self._last = time.time()
        self._warned = False
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(
            __import__('rosgraph_msgs.msg', fromlist=['Clock']).Clock,
            '/clock', self._clock_cb, qos)
        self.pub = self.create_publisher(String, '/sim_stalled', 10)
        self.create_timer(1.0, self._check)
        self.get_logger().info(
            f'Clock watchdog ready (stall timeout {self.stall_timeout:.0f}s)')

    def _clock_cb(self, _msg):
        self._last = time.time()
        if self._warned:
            self.get_logger().info('Simulation clock resumed.')
            self._warned = False

    def _check(self):
        if time.time() - self._last > self.stall_timeout and not self._warned:
            self._warned = True
            self.get_logger().warn(
                f'SIM STALLED: no /clock for {time.time()-self._last:.0f}s')
            self.pub.publish(String(
                data=f'stalled {time.time()-self._last:.0f}s @ {time.time():.0f}'))


def main(args=None):
    rclpy.init(args=args)
    node = ClockWatchdog()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
