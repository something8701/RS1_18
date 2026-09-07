#!/usr/bin/env python3
"""
Husky Survey Patrol: Systematic lawnmower grid with Nav2 obstacle avoidance.
Waits for SLAM + EKF before dispatching goals, and spaces them out.
"""

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Point, Pose, Quaternion
from std_msgs.msg import Header, String
from tf2_ros import Buffer, TransformException
from tf2_ros.transform_listener import TransformListener


class HuskyPatrol(Node):
    """Autonomous lawnmower survey of the forest floor using Nav2."""

    def __init__(self):
        super().__init__('husky_patrol')

        # Survey grid parameters. Defaults start INSIDE the initial SLAM
        # map (~28.5x30.7 m around the spawn): Nav2 rejects goals outside
        # the global costmap, so an off-map first waypoint would keep the
        # robot stationary forever and the map could never grow.
        self.declare_parameter('grid_x_min', -10.0,
            descriptor=ParameterDescriptor(description='Min X grid boundary (m)'))
        self.declare_parameter('grid_x_max', 10.0,
            descriptor=ParameterDescriptor(description='Max X grid boundary (m)'))
        self.declare_parameter('grid_y_min', -10.0,
            descriptor=ParameterDescriptor(description='Min Y grid boundary (m)'))
        self.declare_parameter('grid_y_max', 10.0,
            descriptor=ParameterDescriptor(description='Max Y grid boundary (m)'))
        self.declare_parameter('strip_spacing', 6.0,
            descriptor=ParameterDescriptor(description='Distance between survey strips (m)'))

        x_min = self.get_parameter('grid_x_min').value
        x_max = self.get_parameter('grid_x_max').value
        y_min = self.get_parameter('grid_y_min').value
        y_max = self.get_parameter('grid_y_max').value
        spacing = self.get_parameter('strip_spacing').value

        self.waypoints = []
        y = y_min
        direction = 1
        while y <= y_max:
            x1 = x_min if direction == 1 else x_max
            x2 = x_max if direction == 1 else x_min
            self.waypoints.append((x1, y))
            self.waypoints.append((x2, y))
            y += spacing
            direction *= -1

        self.wp_idx = 0
        self.active = False
        self.goal_in_flight = False
        self._delay_timer = None
        self._mission_active = False

        # TF to check map frame exists
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.nav_client = ActionClient(self, NavigateToPose, '/husky1/navigate_to_pose')

        # Yield Nav2 to the mission coordinator: while it has an active
        # mission (ENROUTE/DISPATCHING), the patrol must not send the next
        # waypoint — a new goal preempts the coordinator's site inspection
        # mid-drive and Nav2 reports it CANCELLED (lost site, no retry).
        self.mission_sub = self.create_subscription(
            String, '/mission_status', self._on_mission_status,
            QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.VOLATILE,
                       history=HistoryPolicy.KEEP_LAST,
                       depth=10)
        )

        self.get_logger().info(
            f'Husky Survey ready: {len(self.waypoints)} waypoints, '
            f'{spacing}m spacing, {x_max-x_min:.0f}x{y_max-y_min:.0f}m'
        )
        # Check every 5s if SLAM + EKF are ready
        self.create_timer(5.0, self._try_start)

    def _try_start(self):
        """Check that SLAM map and odom frames are available before starting."""
        if self.active:
            return

        for frame, label in [('husky1_map', 'SLAM'), ('husky1_odom', 'EKF')]:
            try:
                self.tf_buffer.lookup_transform(
                    frame, 'husky1_base_link', rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=1.0)
                )
            except TransformException:
                self.get_logger().debug(f'Waiting for {frame} ({label})...')
                return

        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().debug('Waiting for Nav2 server...')
            return

        self.active = True
        self.get_logger().info('SLAM + EKF ready. Starting survey.')
        self._send_next()

    def _on_mission_status(self, msg):
        """Track the mission coordinator's state so the patrol can yield."""
        self._mission_active = (
            'ENROUTE' in msg.data or 'DISPATCHING' in msg.data
        )

    def _send_next(self):
        """Send the next waypoint as a Nav2 goal."""
        if self.goal_in_flight:
            return
        if self._mission_active:
            # The mission coordinator is using Nav2 — hold the patrol so
            # the next waypoint does not preempt its site inspection.
            self.get_logger().info(
                'Patrol holding — mission coordinator active.',
                throttle_duration_sec=30.0)
            self._delay_timer = self.create_timer(5.0, self._delayed_send_next)
            return
        if self.wp_idx >= len(self.waypoints):
            self.get_logger().info('Survey complete! Looping...')
            self.wp_idx = 0

        wx, wy = self.waypoints[self.wp_idx]
        self.wp_idx += 1

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped(
            header=Header(
                stamp=self.get_clock().now().to_msg(),
                frame_id='husky1_map',
            ),
            pose=Pose(
                position=Point(x=wx, y=wy, z=0.0),
                orientation=Quaternion(w=1.0),
            ),
        )

        self.get_logger().info(
            f'WP {self.wp_idx}/{len(self.waypoints)}: ({wx:.0f}, {wy:.0f})'
        )

        self.goal_in_flight = True
        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(self._goal_done)

    def _goal_done(self, future):
        """Handle goal acceptance."""
        handle = future.result()
        if handle and handle.accepted:
            handle.get_result_async().add_done_callback(self._result_done)
        else:
            self.get_logger().warn('Goal rejected. Retrying after 5s...')
            self.goal_in_flight = False
            self.wp_idx -= 1  # retry same waypoint
            # Humble note: no one_shot kwarg on create_timer (Iron+ only),
            # so the delay timer is stored and destroyed when it fires.
            self._delay_timer = self.create_timer(5.0, self._delayed_send_next)

    def _result_done(self, future):
        """Handle goal completion."""
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Waypoint reached.')
        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().warn('Waypoint aborted (blocked/unsafe)')
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().warn('Waypoint cancelled')
        else:
            self.get_logger().warn(f'Waypoint status={status}')

        self.goal_in_flight = False
        self._delay_timer = self.create_timer(2.0, self._delayed_send_next)

    def _delayed_send_next(self):
        """One-shot delay between waypoints (Humble-compatible)."""
        if self._delay_timer is not None:
            self.destroy_timer(self._delay_timer)
            self._delay_timer = None
        self._send_next()

    def destroy_node(self):
        self.get_logger().info(
            f'Husky Patrol shutting down. Completed {self.wp_idx}/{len(self.waypoints)} waypoints.'
        )
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HuskyPatrol()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
