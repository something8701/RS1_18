#!/usr/bin/env python3
"""
Demo Patrol Node: Flies the Parrot drone on a survey grid pattern.

Uses direct velocity commands (not Nav2) because the drone flies at 10m
altitude where 2D Nav2 costmaps break. Uses the drone's own odometry
for position feedback and a simple P-controller for waypoint following.

Smart patrol: subscribes to /suspicious_areas. When a flag arrives, the
drone interrupts its patrol, flies to the flagged area, performs a close-up
orbit, then resumes the patrol from where it left off. Flags that arrive
while the drone is already investigating are buffered (latest wins) instead
of being silently dropped.

Refinements over the original:
  - All timing uses the ROS (sim) clock so orbit/dedup windows stay
    consistent under use_sim_time.
  - Publishes /survey_status (STATE, waypoint, coverage %, diversions) so
    the dashboard and decision layer can track survey progress.
  - Per-loop waypoint coverage is tracked and reported.
  - Investigation dedup has a time window (dedup_window) in addition to the
    distance radius.
  - Inside arrival tolerance the drone publishes a zero-linear / gentle
    heading-hold command instead of the last velocity (no drift-through).

Safety: publishes zero cmd_vel on shutdown so the drone doesn't drift.
"""

import math
from collections import deque
from enum import Enum

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from deforestation_interfaces.msg import SuspiciousArea


class PatrolState(Enum):
    PATROL = 'patrol'
    FLYING_TO_FLAG = 'flying_to_flag'
    ORBITING = 'orbiting'
    RETURNING = 'returning'


class DemoPatrol(Node):
    """Flies the Parrot drone in a lawnmower pattern with smart flag response."""

    def __init__(self):
        super().__init__('demo_patrol')

        # Patrol grid
        self.declare_parameter('patrol_x_min', -30.0,
            descriptor=ParameterDescriptor(description='Min X patrol boundary (m)'))
        self.declare_parameter('patrol_x_max', 30.0,
            descriptor=ParameterDescriptor(description='Max X patrol boundary (m)'))
        self.declare_parameter('patrol_y_min', -30.0,
            descriptor=ParameterDescriptor(description='Min Y patrol boundary (m)'))
        self.declare_parameter('patrol_y_max', 30.0,
            descriptor=ParameterDescriptor(description='Max Y patrol boundary (m)'))
        self.declare_parameter('strip_spacing', 8.0,
            descriptor=ParameterDescriptor(description='Distance between survey strips (m)'))
        self.declare_parameter('altitude', 10.0,
            descriptor=ParameterDescriptor(description='Drone patrol altitude (m) — parrot spawns at z=10'))
        self.declare_parameter('arrival_tolerance', 1.5,
            descriptor=ParameterDescriptor(description='Waypoint arrival radius (m)'))
        self.declare_parameter('max_speed', 3.0,
            descriptor=ParameterDescriptor(description='Maximum flight speed (m/s)'))
        self.declare_parameter('k_linear', 2.0,
            descriptor=ParameterDescriptor(description='Linear P-controller gain'))
        self.declare_parameter('k_angular', 2.0,
            descriptor=ParameterDescriptor(description='Angular P-controller gain'))
        # Attitude safety. The Parrot is moved by VelocityControl with gravity
        # off and no attitude controller, so a tilt never corrects itself; fast
        # yaw during flag orbits has pitched it over and flown it into the
        # ground.
        self.declare_parameter('max_yaw_rate', 1.5,
            descriptor=ParameterDescriptor(description='Yaw rate limit (rad/s)'))
        self.declare_parameter('k_level', 2.0,
            descriptor=ParameterDescriptor(
                description='Roll/pitch levelling gain: angular.x/y = -k * roll/pitch'))
        self.declare_parameter('max_tilt_deg', 30.0,
            descriptor=ParameterDescriptor(
                description='Above this roll/pitch, forward speed is zeroed until level'))
        # Smart patrol params
        self.declare_parameter('enable_smart_patrol', True,
            descriptor=ParameterDescriptor(description='Enable reactive patrol (interrupt on flag)'))
        self.declare_parameter('investigation_altitude', 5.0,
            descriptor=ParameterDescriptor(description='Altitude during close-up investigation (m)'))
        self.declare_parameter('orbit_radius', 8.0,
            descriptor=ParameterDescriptor(description='Orbit radius around flag (m)'))
        self.declare_parameter('orbit_duration', 15.0,
            descriptor=ParameterDescriptor(description='Time to orbit the flag (s)'))
        # Refinements
        self.declare_parameter('dedup_radius', 10.0,
            descriptor=ParameterDescriptor(description='Skip flags within this radius (m) '
                            'of a recent investigation'))
        self.declare_parameter('dedup_window', 300.0,
            descriptor=ParameterDescriptor(description='Time window (s, sim clock) for '
                            'investigation dedup — older entries are ignored'))
        self.declare_parameter('status_period', 2.0,
            descriptor=ParameterDescriptor(description='Seconds between /survey_status publishes'))

        self.x_min = self.get_parameter('patrol_x_min').value
        self.x_max = self.get_parameter('patrol_x_max').value
        self.y_min = self.get_parameter('patrol_y_min').value
        self.y_max = self.get_parameter('patrol_y_max').value
        self.spacing = self.get_parameter('strip_spacing').value
        self.altitude = self.get_parameter('altitude').value
        self.arrival_tolerance = self.get_parameter('arrival_tolerance').value
        self.max_speed = self.get_parameter('max_speed').value
        self.k_linear = self.get_parameter('k_linear').value
        self.k_angular = self.get_parameter('k_angular').value
        self.max_yaw_rate = float(self.get_parameter('max_yaw_rate').value)
        self.k_level = float(self.get_parameter('k_level').value)
        self.max_tilt = math.radians(float(self.get_parameter('max_tilt_deg').value))
        self.roll = 0.0
        self.pitch = 0.0
        self._tilt_warned = False
        self.smart_patrol_enabled = self.get_parameter('enable_smart_patrol').value
        self.investigation_alt = self.get_parameter('investigation_altitude').value
        self.orbit_radius = self.get_parameter('orbit_radius').value
        self.orbit_duration = self.get_parameter('orbit_duration').value
        self.dedup_radius = self.get_parameter('dedup_radius').value
        self.dedup_window = self.get_parameter('dedup_window').value
        self.status_period = self.get_parameter('status_period').value

        # Build lawnmower waypoints
        self.waypoints = []
        y = self.y_min
        direction = 1
        while y <= self.y_max:
            x1 = self.x_min if direction == 1 else self.x_max
            x2 = self.x_max if direction == 1 else self.x_min
            self.waypoints.append((x1, y))
            self.waypoints.append((x2, y))
            y += self.spacing
            direction *= -1

        self.current_wp = 0
        self.saved_wp = 0  # where to resume after investigation
        self._visited_wps = set()  # waypoints reached in the current loop

        # State machine
        self.state = PatrolState.PATROL
        self.orbit_start_time = 0.0
        self.orbit_logged_sec = -1
        self.alt_warned = False
        self.flag_target_x = 0.0
        self.flag_target_y = 0.0
        self.pending_flag = None  # buffered flag while busy (latest wins)
        self.divert_count = 0

        # Recent flags for dedup: (x, y, sim-clock seconds)
        self._recent_investigations = deque(maxlen=20)

        # Current pose
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self._pose_received = False

        # -- QoS --
        cmd_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        alert_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Publisher
        self.cmd_pub = self.create_publisher(Twist, '/parrot1/cmd_vel', cmd_qos)
        self.status_pub = self.create_publisher(String, '/survey_status', alert_qos)

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/parrot1/odometry', self.odom_callback, sensor_qos)
        self.flag_sub = self.create_subscription(
            SuspiciousArea, '/suspicious_areas', self.flag_callback, alert_qos)

        # Control loop at 20Hz
        self.timer = self.create_timer(0.05, self.control_loop)
        self.status_timer = self.create_timer(self.status_period, self._publish_status)

        self.get_logger().info(
            f'Patrol ready: {len(self.waypoints)} waypoints, '
            f'{self.spacing}m spacing, {self.altitude}m altitude. '
            f'Smart patrol: {"ON" if self.smart_patrol_enabled else "OFF"}.'
        )

    # ── Helpers ──────────────────────────────────────────────────────

    def _now_s(self) -> float:
        """Current ROS (sim) time in seconds."""
        return self.get_clock().now().nanoseconds / 1e9

    # ── Odometry ──────────────────────────────────────────────────────

    def odom_callback(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny, cosy)
        self.roll = math.atan2(2.0 * (q.w * q.x + q.y * q.z),
                               1.0 - 2.0 * (q.x * q.x + q.y * q.y))
        self.pitch = math.asin(max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x))))
        self._pose_received = True

    def _publish(self, cmd: Twist):
        """Publish cmd_vel with attitude hold and safety limits."""
        cmd.angular.z = max(-self.max_yaw_rate,
                            min(self.max_yaw_rate, cmd.angular.z))
        cmd.angular.x = -self.k_level * self.roll
        cmd.angular.y = -self.k_level * self.pitch
        tilted = max(abs(self.roll), abs(self.pitch)) > self.max_tilt
        if tilted:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            if not self._tilt_warned:
                self.get_logger().error(
                    f'Drone tilted (roll {math.degrees(self.roll):.0f} deg, '
                    f'pitch {math.degrees(self.pitch):.0f} deg): holding '
                    f'position and levelling.')
                self._tilt_warned = True
        elif self._tilt_warned:
            self.get_logger().info('Drone level again, resuming.')
            self._tilt_warned = False
        self.cmd_pub.publish(cmd)

    # ── Smart patrol: flag reaction ───────────────────────────────────

    def flag_callback(self, msg: SuspiciousArea):
        """React to a suspicious area flag.

        Idle (PATROL) → divert immediately. Busy → buffer the flag (latest
        wins) and investigate once the current task completes.
        """
        if not self.smart_patrol_enabled:
            return

        fx = msg.position.x
        fy = msg.position.y
        now = self._now_s()

        # Dedup: don't re-investigate areas within dedup_radius AND within
        # dedup_window of a recent investigation.
        for prev_x, prev_y, prev_t in self._recent_investigations:
            if (now - prev_t) < self.dedup_window and \
                    math.hypot(fx - prev_x, fy - prev_y) < self.dedup_radius:
                self.get_logger().debug(
                    f'Flag at ({fx:.1f},{fy:.1f}) too close to recent investigation, skipping.')
                return

        self._recent_investigations.append((fx, fy, now))

        if self.state != PatrolState.PATROL:
            self.pending_flag = (fx, fy)
            self.get_logger().info(
                f'FLAG BUFFERED: {msg.type} at ({fx:.1f},{fy:.1f}) — '
                f'busy ({self.state.value}), investigating after current task.')
            return

        self._begin_investigation(fx, fy)
        self.get_logger().warn(
            f'FLAG INTERRUPT: {msg.type} at ({fx:.1f},{fy:.1f}). '
            f'Diverting from WP {self.current_wp}/{len(self.waypoints)}.'
        )

    def _begin_investigation(self, fx: float, fy: float):
        """Switch from patrol to flying toward a flag target."""
        # Remember where the patrol was so it can resume here afterwards
        # (otherwise _control_return flies back to waypoint 0 every time).
        self.saved_wp = self.current_wp
        self.divert_count += 1
        self.flag_target_x = fx
        self.flag_target_y = fy
        self.state = PatrolState.FLYING_TO_FLAG

    # ── Main control loop ─────────────────────────────────────────────

    def control_loop(self):
        if not self._pose_received:
            return

        if self.state == PatrolState.PATROL:
            self._control_patrol()
        elif self.state == PatrolState.FLYING_TO_FLAG:
            self._control_fly_to_flag()
        elif self.state == PatrolState.ORBITING:
            self._control_orbit()
        elif self.state == PatrolState.RETURNING:
            self._control_return()

    def _control_patrol(self):
        """Standard lawnmower waypoint following."""
        if self.current_wp >= len(self.waypoints):
            self.get_logger().info('Patrol complete. Looping...')
            self.current_wp = 0
            self._visited_wps.clear()

        tx, ty = self.waypoints[self.current_wp]
        self._fly_to(tx, ty, self.arrival_tolerance, self.max_speed)

        dist = math.hypot(tx - self.x, ty - self.y)
        if dist < self.arrival_tolerance:
            self._visited_wps.add(self.current_wp)
            self.current_wp += 1
            if self.current_wp < len(self.waypoints):
                nx, ny = self.waypoints[self.current_wp]
                self.get_logger().info(
                    f'WP {self.current_wp}/{len(self.waypoints)}: ({nx:.1f}, {ny:.1f})')

    def _control_fly_to_flag(self):
        """Fly to the flagged area for close inspection."""
        dist = math.hypot(self.flag_target_x - self.x, self.flag_target_y - self.y)
        self._fly_to(self.flag_target_x, self.flag_target_y,
                     self.arrival_tolerance, self.max_speed * 0.7)

        if dist < self.arrival_tolerance:
            if not self.alt_warned:
                # cmd_vel is 2D and odometry publishes z=0, so the altitude
                # cannot change; the patrol altitude is held.
                self.get_logger().warn(
                    f'investigation_altitude={self.investigation_alt}m is not '
                    f'applied: 2D cmd_vel cannot change altitude in this sim. '
                    f'Orbiting at patrol altitude.')
                self.alt_warned = True
            self.get_logger().info(
                f'Arrived at flag ({self.flag_target_x:.1f},{self.flag_target_y:.1f}). '
                f'Starting orbit at {self.orbit_radius}m radius...')
            self.state = PatrolState.ORBITING
            self.orbit_start_time = self._now_s()

    def _control_orbit(self):
        """Circle the flagged area at orbit_radius for close inspection."""
        elapsed = self._now_s() - self.orbit_start_time
        if elapsed > self.orbit_duration:
            self.get_logger().info('Orbit complete.')
            self._take_pending_or(self._start_return)
            return

        # Orbit: fly tangent to the circle, biased inward/outward by the
        # radius error so the circle converges to orbit_radius.
        dx = self.x - self.flag_target_x
        dy = self.y - self.flag_target_y
        dist_to_center = math.hypot(dx, dy)

        if dist_to_center < 0.5:
            # Too close to the flag to define a tangent: hold a gentle turn.
            tangent_angle = self.yaw
            bias = 0.0
        else:
            tangent_angle = math.atan2(dx, -dy)  # counter-clockwise tangent
            radial_error = dist_to_center - self.orbit_radius
            bias = math.atan2(-radial_error * 0.5, self.orbit_radius)

        desired_yaw = tangent_angle + bias
        yaw_error = math.atan2(math.sin(desired_yaw - self.yaw),
                               math.cos(desired_yaw - self.yaw))

        cmd = Twist()
        cmd.linear.x = self.max_speed * 0.5
        cmd.angular.z = self.k_angular * yaw_error
        self._publish(cmd)

        sec = int(elapsed)
        if sec != self.orbit_logged_sec:
            self.orbit_logged_sec = sec
            self.get_logger().debug(
                f'Orbiting... {elapsed:.0f}s / {self.orbit_duration:.0f}s, '
                f'r={dist_to_center:.1f}m (target={self.orbit_radius}m)')

    def _take_pending_or(self, on_none):
        """End-of-task decision: divert to a buffered flag, else call on_none."""
        if self.pending_flag is not None:
            fx, fy = self.pending_flag
            self.pending_flag = None
            self.get_logger().info(
                f'Taking buffered flag at ({fx:.1f},{fy:.1f}).')
            self._begin_investigation(fx, fy)
            return
        on_none()

    def _start_return(self):
        self.get_logger().info('Returning to patrol...')
        self.state = PatrolState.RETURNING

    def _resume_patrol(self):
        self.current_wp = self.saved_wp
        self.state = PatrolState.PATROL
        self.get_logger().info(f'Resumed patrol at WP {self.saved_wp}.')

    def _control_return(self):
        """Fly back to the saved patrol waypoint and resume patrol."""
        if self.saved_wp >= len(self.waypoints):
            self.saved_wp = 0

        tx, ty = self.waypoints[self.saved_wp]
        dist = math.hypot(tx - self.x, ty - self.y)
        self._fly_to(tx, ty, self.arrival_tolerance * 2, self.max_speed)

        if dist < self.arrival_tolerance * 2:
            self._take_pending_or(self._resume_patrol)

    def _fly_to(self, tx, ty, tolerance, max_spd):
        """P-controller to fly toward (tx, ty)."""
        dx = tx - self.x
        dy = ty - self.y
        dist = math.hypot(dx, dy)

        cmd = Twist()
        if dist < tolerance:
            # Hold position: zero linear, gentle heading correction toward
            # the target so the drone does not keep its last velocity or
            # rotate away while the controller waits for the state change.
            target_yaw = math.atan2(dy, dx)
            yaw_error = target_yaw - self.yaw
            yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))
            cmd.linear.x = 0.0
            cmd.angular.z = 0.3 * yaw_error
            self._publish(cmd)
            return

        target_yaw = math.atan2(dy, dx)
        yaw_error = target_yaw - self.yaw
        yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))

        cmd.linear.x = min(self.k_linear * dist, max_spd)
        cmd.angular.z = self.k_angular * yaw_error
        self._publish(cmd)

    # ── Survey status (feeds the dashboard + decision layer) ──────────

    def _publish_status(self):
        if self.state == PatrolState.PATROL:
            coverage = len(self._visited_wps) / max(len(self.waypoints), 1) * 100.0
            status = (
                f'STATE=PATROL wp={min(self.current_wp + 1, len(self.waypoints))}/'
                f'{len(self.waypoints)} coverage={coverage:.0f}% '
                f'diverted={self.divert_count}'
            )
        elif self.state == PatrolState.FLYING_TO_FLAG:
            status = (f'STATE=FLYING_TO_FLAG '
                      f'target=({self.flag_target_x:.1f},{self.flag_target_y:.1f}) '
                      f'diverted={self.divert_count}')
        elif self.state == PatrolState.ORBITING:
            elapsed = self._now_s() - self.orbit_start_time
            status = (f'STATE=ORBITING '
                      f'target=({self.flag_target_x:.1f},{self.flag_target_y:.1f}) '
                      f'elapsed={elapsed:.0f}s/{self.orbit_duration:.0f}s')
        else:  # RETURNING
            status = (f'STATE=RETURNING wp={self.saved_wp} '
                      f'diverted={self.divert_count}')
        self.status_pub.publish(String(data=status))

    def destroy_node(self):
        self.get_logger().info(
            f'Stopping patrol. State was: {self.state.value}. Sending zero velocity...')
        stop = Twist()
        self.cmd_pub.publish(stop)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DemoPatrol()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
