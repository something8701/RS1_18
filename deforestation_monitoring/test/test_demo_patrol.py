"""demo_patrol cmd_vel safety: attitude hold, yaw-rate clamp, tilt stop."""

import math

import pytest
import rclpy
from geometry_msgs.msg import Twist

from deforestation_monitoring.demo_patrol import DemoPatrol


@pytest.fixture(scope="module")
def ros_ctx():
    rclpy.init()
    yield
    rclpy.shutdown()


def _capture(node):
    sent = []
    node.cmd_pub.publish = sent.append
    return sent


def test_levelling_and_yaw_clamp(ros_ctx):
    node = DemoPatrol()
    sent = _capture(node)
    node.roll, node.pitch = math.radians(5), math.radians(-8)
    cmd = Twist()
    cmd.linear.x = 3.0
    cmd.angular.z = 6.0                      # orbit P-term can reach ~2*pi
    node._publish(cmd)
    out = sent[-1]
    assert out.linear.x == 3.0
    assert out.angular.z == pytest.approx(node.max_yaw_rate)
    assert out.angular.x < 0 < out.angular.y   # opposes roll / pitch
    node.destroy_node()


def test_tilted_drone_stops_forward_motion(ros_ctx):
    """A drone pitched 84 degrees that is still told to fly 3 m/s forward
    flies into the ground."""
    node = DemoPatrol()
    sent = _capture(node)
    node.roll, node.pitch = 0.0, math.radians(84)
    cmd = Twist()
    cmd.linear.x = 3.0
    node._publish(cmd)
    assert sent[-1].linear.x == 0.0
    assert sent[-1].angular.y < 0
    node.destroy_node()
