#!/usr/bin/env python3
"""
Mission coordinator: takes SuspiciousArea flags and sends the Husky to
inspect each site with Nav2 NavigateToPose.

Priority modes:
  - closest:    nearest to the robot first (from /husky1/odometry)
  - confidence: highest confidence first
  - area:       largest area first
  - severity:   highest severity first
  - composite:  (severity * confidence) / distance, so big, certain and
                nearby sites go first

Other behaviour:
  - closest and composite are re-ranked on every queue tick as the robot
    moves.
  - Missions with equal priority run in arrival order.
  - A site is skipped if it is near a queued or inspected site and inside
    dedup_window.
  - Rejected or aborted goals are queued again with a priority penalty, up
    to max_retries.
  - The latest camera image is kept from a normal subscription and saved as
    evidence on arrival.
  - Nav2 readiness is checked without blocking the queue timer.
  - /mission_status is published for the dashboard.
  - Active Nav2 goals are cancelled on shutdown.
"""

import cv2
import heapq
import math
import os
import csv
import time
from collections import deque
from dataclasses import dataclass, field

import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Point, Pose, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header, String
from deforestation_interfaces.msg import SuspiciousArea
from cv_bridge import CvBridge, CvBridgeError
from tf2_ros import Buffer, TransformException, TransformListener


@dataclass(order=True)
class Mission:
    """An inspection task. A lower priority value is more urgent.

    Tasks compare by (priority, seq), so equal priorities run in arrival
    order. Fields with defaults are last so positional construction still
    works.
    """
    priority: float
    x: float = field(compare=False)
    y: float = field(compare=False)
    type: str = field(compare=False)
    confidence: float = field(compare=False)
    area_m2: float = field(compare=False)
    severity: float = field(compare=False)
    label: str = field(compare=False)
    frame_id: str = field(compare=False)
    seq: int = field(default=0)
    attempts: int = field(default=0, compare=False)


class MissionCoordinator(Node):
    """Takes SuspiciousArea flags and sends the Husky with Nav2."""

    def __init__(self):
        super().__init__('mission_coordinator')

        self.latest_scan = None
        self._lidar_subscription = self.create_subscription(
            LaserScan,
            '/husky1/scan',
            self._lidar_callback,
            qos_profile_sensor_data
        )

        self.declare_parameter('priority_mode', 'composite',
            descriptor=ParameterDescriptor(
                description='Priority mode: closest, confidence, area, severity, composite'))
        self.declare_parameter('capture_images', True,
            descriptor=ParameterDescriptor(
                description='Capture camera image on arrival at flagged site'))
        self.declare_parameter('image_dir', '/tmp/deforestation_inspections',
            descriptor=ParameterDescriptor(
                description='Directory to save inspection images'))
        self.declare_parameter('map_frame', 'husky1_map',
            descriptor=ParameterDescriptor(
                description='Nav2 global frame; flags are transformed into it'))
        self.declare_parameter('dedup_radius', 10.0,
            descriptor=ParameterDescriptor(
                description='Skip flags within this radius (m) of a queued or '
                            'inspected site, so repeated detections do not flood the queue'))
        self.declare_parameter('dedup_window', 300.0,
            descriptor=ParameterDescriptor(
                description='Time window (s, sim clock) for skipping a site. '
                            'After it, the site can be queued again'))
        self.declare_parameter('max_retries', 3,
            descriptor=ParameterDescriptor(
                description='Max Nav2 goal retries for rejected/aborted missions'))
        self.declare_parameter('requeue_penalty', 50.0,
            descriptor=ParameterDescriptor(
                description='Priority penalty added to a mission on each retry'))

        self.declare_parameter('inspection_standoff', 1.0,
            descriptor=ParameterDescriptor(
                description='Distance (m) the Husky stops short of a suspicious area'))

        self.priority_mode = self.get_parameter('priority_mode').value
        self.capture_enabled = self.get_parameter('capture_images').value
        self.image_dir = self.get_parameter('image_dir').value
        self.map_frame = self.get_parameter('map_frame').value
        self.dedup_radius = self.get_parameter('dedup_radius').value
        self.dedup_window = self.get_parameter('dedup_window').value
        self.max_retries = self.get_parameter('max_retries').value
        self.requeue_penalty = self.get_parameter('requeue_penalty').value
        self.inspection_standoff = self.get_parameter('inspection_standoff').value

        # TF, to transform flag positions into the Nav2 map frame
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Ensure image directory exists
        if self.capture_enabled:
            os.makedirs(self.image_dir, exist_ok=True)

        self.mission_queue = []
        self.current_mission = None
        self.active = False
        self.inspected = 0
        self.report_seq = 0
        self.total = 0
        self._seq_counter = 0
        self._goal_handle = None

        # Robot pose (unknown until first odometry message)
        self.robot_x = None
        self.robot_y = None

        # Recently queued or inspected sites: (x, y, sim-clock ns)
        self._recent_sites = deque(maxlen=50)

        # Latest camera image, from a normal subscription
        self.bridge = CvBridge()
        self._latest_image = None

        # QoS profiles
        alert_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # Subscribers
        self.flag_sub = self.create_subscription(
            SuspiciousArea, '/suspicious_areas', self.flag_callback, alert_qos
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/husky1/odometry', self.odom_callback, sensor_qos
        )
        self.cam_sub = self.create_subscription(
            Image, '/husky1/camera/image', self.cam_callback, sensor_qos
        )
        self.report_pub = self.create_publisher(
            String, '/inspection_reports', alert_qos
        )
        self.status_pub = self.create_publisher(
            String, '/mission_status', alert_qos
        )

        # Nav2 action client
        self.nav_client = ActionClient(
            self, NavigateToPose, '/husky1/navigate_to_pose'
        )

        self.create_timer(2.0, self.process_queue)
        self._publish_status()
        self.get_logger().info(
            f'Mission Coordinator ready. Priority mode: {self.priority_mode}. '
            f'Camera capture: {self.capture_enabled}.'
        )

    # Priority scoring

    def _compute_priority_value(self, mode: str, severity: float,
                                confidence: float, area_m2: float,
                                dist: float) -> float:
        """Return the priority value of a flag. Smaller is more urgent."""
        if mode == 'closest':
            # Closest first
            return dist

        elif mode == 'confidence':
            # Highest confidence first (negated, since smaller is more urgent)
            return -confidence

        elif mode == 'area':
            # Largest area first
            return -area_m2

        elif mode == 'severity':
            # Highest severity first
            return -severity

        elif mode == 'composite':
            # (severity * confidence) / (distance + 1), negated, so close,
            # certain and large threats come first. The +1 avoids dividing
            # by zero.
            score = (severity * confidence) / (dist + 1.0)
            return -score

        else:
            self.get_logger().warn(
                f'Unknown priority_mode: {mode}, using closest',
                throttle_duration_sec=30.0)
            return dist

    def _distance_to(self, gx: float, gy: float) -> float:
        """Distance from the Husky to a flag.

        Uses the map origin until the first Husky odometry arrives.
        """
        if self.robot_x is not None:
            return math.hypot(gx - self.robot_x, gy - self.robot_y)
        self.get_logger().warn(
            'No Husky odometry yet — scoring distance from map origin.',
            throttle_duration_sec=10.0)
        return math.hypot(gx, gy)

    # Flag handling

    def odom_callback(self, msg: Odometry):
        """Store the Husky pose for distance-based priorities."""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

    def cam_callback(self, msg: Image):
        """Keep the latest camera frame for inspection evidence."""
        self._latest_image = msg

    def flag_callback(self, msg: SuspiciousArea):
        """Queue a new suspicious area for inspection."""
        gx = msg.position.x
        gy = msg.position.y
        now_ns = self.get_clock().now().nanoseconds

        # Skip a site that is within dedup_radius of a recent one and inside
        # dedup_window. Older inspections expire, so a site can be inspected
        # again later.
        for sx, sy, st in self._recent_sites:
            if (now_ns - st) < self.dedup_window * 1e9 and \
                    math.hypot(gx - sx, gy - sy) < self.dedup_radius:
                self.get_logger().info(
                    f'Flag @ ({gx:.1f},{gy:.1f}) within {self.dedup_radius:.0f}m '
                    f'of a recent site — skipping duplicate.',
                    throttle_duration_sec=5.0)
                return

        dist = self._distance_to(gx, gy)

        self.total += 1
        self._recent_sites.append((gx, gy, now_ns))

        # Priority of this flag with the selected mode
        priority = self._compute_priority_value(
            self.priority_mode, msg.severity, msg.confidence,
            msg.area_m2, dist)

        heapq.heappush(self.mission_queue, Mission(
            priority=priority,
            seq=self._seq_counter,
            x=gx, y=gy,
            type=msg.type,
            confidence=msg.confidence,
            area_m2=msg.area_m2,
            severity=msg.severity,
            label=(
                f'[{msg.type}] S={msg.severity:.0f} '
                f'C={msg.confidence:.0f}% {msg.area_m2:.0f}m²'
            ),
            frame_id=msg.header.frame_id,
        ))
        self._seq_counter += 1

        self.get_logger().info(
            f'Flag #{self.total} queued @ ({gx:.1f},{gy:.1f}) '
            f'({dist:.1f}m from robot): {msg.type} '
            f'sev={msg.severity:.0f} conf={msg.confidence:.0f}% '
            f'area={msg.area_m2:.0f}m² | Queue: {len(self.mission_queue)} '
            f'(mode={self.priority_mode})'
        )
        self._publish_status()

    def process_queue(self):
        """Start the next mission in the queue if the robot is idle."""
        if self.active or not self.mission_queue:
            return

        # Check that Nav2 is ready without blocking. The action server is a
        # lifecycle node and may still be starting.
        if not self.nav_client.server_is_ready():
            self.get_logger().warn(
                'Nav2 server not ready — waiting.', throttle_duration_sec=5.0)
            return

        # Distance-based modes: re-rank the queue from the current Husky
        # pose, not the pose when each flag was queued.
        if self.robot_x is not None and \
                self.priority_mode in ('closest', 'composite'):
            re_scored = []
            while self.mission_queue:
                m = heapq.heappop(self.mission_queue)
                dist = math.hypot(m.x - self.robot_x, m.y - self.robot_y)
                m.priority = self._compute_priority_value(
                    self.priority_mode, m.severity, m.confidence,
                    m.area_m2, dist)
                re_scored.append(m)
            for m in re_scored:
                heapq.heappush(self.mission_queue, m)

        mission = heapq.heappop(self.mission_queue)
        self.current_mission = mission
        self.active = True
        self._publish_status()
        self._send_goal(mission)

    # Navigation

    def _requeue(self, mission: Mission, penalty: float):
        """Queue a mission again with a priority penalty and a retry count."""
        mission.priority += penalty
        mission.attempts += 1
        heapq.heappush(self.mission_queue, mission)
        self.get_logger().warn(
            f'Requeued {mission.label} (attempt {mission.attempts}/'
            f'{self.max_retries}) with priority penalty {penalty:.0f}. '
            f'Queue: {len(self.mission_queue)}')

    def _send_goal(self, mission: Mission):
        """Send a NavigateToPose goal to Nav2.

        Flags may be in another frame (for example parrot1_odom from the
        drone scan mapper). The position is transformed into the Nav2 map
        frame with TF when possible. If TF has no path, the frames are taken
        to be the same (both start at the sim world origin).
        """
        if not self.nav_client.server_is_ready():
            self.get_logger().warn('Nav2 server went away — requeuing mission.')
            self._requeue(mission, self.requeue_penalty)
            self.active = False
            self._publish_status()
            return

        gx, gy, gframe = mission.x, mission.y, mission.frame_id
        if gframe and gframe != self.map_frame:
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.map_frame, gframe, rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.2)
                )
                t = tf.transform.translation
                q = tf.transform.rotation
                # Rotate the flag offset about z only (ground plane)
                siny = 2.0 * (q.w * q.z + q.x * q.y)
                cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                yaw = math.atan2(siny, cosy)
                cy, sy = math.cos(yaw), math.sin(yaw)
                mx = t.x + gx * cy - gy * sy
                my = t.y + gx * sy + gy * cy
                gx, gy = mx, my
                gframe = self.map_frame
            except TransformException:
                self.get_logger().warn(
                    f'No TF {gframe} → {self.map_frame}; assuming frames coincide '
                    f'(using raw flag coordinates)',
                    throttle_duration_sec=10.0)

        # Stop inspection_standoff metres short of the suspicious area
        inspection_standoff = self.inspection_standoff

        try:
            robot_tf = self.tf_buffer.lookup_transform(
                self.map_frame,
                'husky1_base_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2)
            )
            rx = robot_tf.transform.translation.x
            ry = robot_tf.transform.translation.y
        except TransformException:
            if self.robot_x is not None and self.robot_y is not None:
                rx = self.robot_x
                ry = self.robot_y
            else:
                rx = 0.0
                ry = 0.0

        dx = gx - rx
        dy = gy - ry
        distance = math.hypot(dx, dy)

        if distance > inspection_standoff:
            goal_x = gx - (dx / distance) * inspection_standoff
            goal_y = gy - (dy / distance) * inspection_standoff
        else:
            goal_x = rx
            goal_y = ry

        inspection_yaw = math.atan2(gy - goal_y, gx - goal_x)
        goal_orientation = Quaternion(
            z=math.sin(inspection_yaw / 2.0),
            w=math.cos(inspection_yaw / 2.0),
        )

        self.get_logger().info(
            f'Inspection pose: ({goal_x:.2f},{goal_y:.2f}) '
            f'for alert ({gx:.2f},{gy:.2f})'
        )

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped(
            header=Header(
                stamp=self.get_clock().now().to_msg(),
                frame_id=gframe,
            ),
            pose=Pose(
                position=Point(x=goal_x, y=goal_y, z=0.0),
                orientation=goal_orientation,
            ),
        )

        self.get_logger().info(
            f'Dispatching Husky → {mission.label} (mode={self.priority_mode})'
        )
        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(self._goal_response)

    def _goal_response(self, future):
        """Handle Nav2 goal acceptance."""
        handle = future.result()
        if handle is None or not handle.accepted:
            self.get_logger().warn('Goal rejected by Nav2')
            m = self.current_mission
            if m is not None and m.attempts < self.max_retries:
                self._requeue(m, self.requeue_penalty)
            elif m is not None:
                self.report_seq += 1
                msg = f'SITE UNREACHABLE #{self.report_seq}: {m.label} @ ({m.x:.1f},{m.y:.1f})'
                self.get_logger().warn(msg)
                self.report_pub.publish(String(data=msg))
            self.current_mission = None
            self.active = False
            self._publish_status()
            return
        self._goal_handle = handle
        self.get_logger().info('Goal accepted. Husky en route.')
        handle.get_result_async().add_done_callback(self._result)

    def _result(self, future):
        """Handle the Nav2 result and save evidence on success."""
        self._goal_handle = None
        result = future.result()
        status = result.status
        m = self.current_mission

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.inspected += 1
            self.report_seq += 1
            evidence_path = self._capture_evidence()
            evidence_str = f' [image: {evidence_path}]' if evidence_path else ' [no image]'
            self._save_lidar_evidence(evidence_path)
            classification, reason = self._classify_disturbance(m, evidence_path)

            msg = (
                f'SITE INSPECTED #{self.report_seq}: {m.label} '
                f'@ ({m.x:.1f},{m.y:.1f}) [{classification}]{evidence_str}'
            )
            self.get_logger().info(msg)
            self.get_logger().info(f'  decision: {classification} - {reason}')

        elif status == GoalStatus.STATUS_CANCELED:
            self.report_seq += 1
            msg = f'SITE CANCELLED #{self.report_seq}: {m.label} @ ({m.x:.1f},{m.y:.1f})'
            self.get_logger().warn(msg)

        elif status == GoalStatus.STATUS_ABORTED:
            if m.attempts < self.max_retries:
                # Temporary failure (stuck path, old costmap): queue again
                # with a priority penalty instead of dropping the site.
                self.get_logger().warn(
                    f'SITE RETRY: {m.label} aborted — requeuing.')
                self._requeue(m, self.requeue_penalty)
                self.current_mission = None
                self.active = False
                self._publish_status()
                return
            self.report_seq += 1
            msg = f'SITE UNREACHABLE #{self.report_seq}: {m.label} @ ({m.x:.1f},{m.y:.1f})'
            self.get_logger().warn(msg)

        else:
            self.report_seq += 1
            msg = f'SITE STATUS={status} #{self.report_seq}: {m.label} @ ({m.x:.1f},{m.y:.1f})'
            self.get_logger().warn(msg)

        self.report_pub.publish(String(data=msg))
        self.current_mission = None
        self.active = False
        self._publish_status()

    # Evidence capture

    def _capture_evidence(self) -> str:
        """Save the latest Husky camera image as inspection evidence.

        Uses the stored latest image from the camera subscription.

        Returns the saved file path, or an empty string on failure.
        """
        if not self.capture_enabled:
            return ''

        if self._latest_image is None:
            self.get_logger().warn('No camera image received yet for evidence capture')
            return ''

        try:
            frame = self.bridge.imgmsg_to_cv2(self._latest_image, 'bgr8')
        except CvBridgeError:
            self.get_logger().warn('Failed to convert camera image for evidence')
            return ''

        ts = time.strftime('%Y%m%d_%H%M%S')
        filename = f'inspection_{self.inspected:03d}_{ts}.jpg'
        filepath = os.path.join(self.image_dir, filename)
        cv2.imwrite(filepath, frame)
        self.get_logger().info(f'Evidence saved: {filepath}')
        return filepath

    def _lidar_callback(self, msg):
        """Store the latest Husky laser scan for inspection evidence."""
        self.latest_scan = msg

    def _save_lidar_evidence(self, image_path=''):
        """Save the latest Husky LiDAR scan beside the camera evidence."""
        scan = self.latest_scan

        if scan is None:
            self.get_logger().warning(
                'No LiDAR scan available for this inspection.'
            )
            return ''

        if image_path:
            lidar_path = os.path.splitext(image_path)[0] + '_lidar.csv'
        else:
            evidence_dir = '/tmp/deforestation_inspections'
            os.makedirs(evidence_dir, exist_ok=True)

            stamp = int(self.get_clock().now().nanoseconds / 1e9)
            lidar_path = os.path.join(
                evidence_dir,
                f'inspection_{stamp}_lidar.csv'
            )

        valid_points = 0
        angle = scan.angle_min

        with open(lidar_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'angle_rad',
                'range_m',
                'x_m',
                'y_m'
            ])

            for distance in scan.ranges:
                if (
                    math.isfinite(distance)
                    and scan.range_min <= distance <= scan.range_max
                ):
                    x = distance * math.cos(angle)
                    y = distance * math.sin(angle)

                    writer.writerow([
                        angle,
                        distance,
                        x,
                        y
                    ])

                    valid_points += 1

                angle += scan.angle_increment

        self.get_logger().info(
            f'LiDAR evidence saved: {lidar_path} '
            f'({valid_points} valid points)'
        )

        return lidar_path

    def _classify_disturbance(self, mission, evidence_path=''):
        """Use the Husky camera image to support the inspection result."""

        if not evidence_path or not os.path.exists(evidence_path):
            return 'UNCERTAIN', 'no usable camera evidence available'

        frame = cv2.imread(evidence_path)
        if frame is None:
            return 'UNCERTAIN', 'camera evidence could not be loaded'

        # Find strong straight edges in the camera image.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)

        h, w = gray.shape[:2]
        min_line_length = max(30, int(0.15 * min(h, w)))

        lines = cv2.HoughLinesP(
            edges,
            1,
            math.pi / 180.0,
            threshold=50,
            minLineLength=min_line_length,
            maxLineGap=20
        )

        line_count = 0 if lines is None else len(lines)
        edge_ratio = cv2.countNonZero(edges) / float(edges.size)

        self.get_logger().info(
            f'Camera evidence metrics: '
            f'long_lines={line_count}, edge_ratio={edge_ratio:.3f}'
        )

        alert_type = (mission.type or '').upper()

        # A LINE alert needs straight-edge evidence before it is classified
        # as deliberate cutting.
        if alert_type == 'LINE':
            if line_count >= 3:
                return (
                    'CUT',
                    f'camera evidence detected {line_count} strong straight edges '
                    f'(edge ratio {edge_ratio:.3f})'
                )

            return (
                'UNCERTAIN',
                f'LINE alert received but camera found only {line_count} '
                f'strong straight edges'
            )

        if alert_type == 'GAP':
            # A gap with few straight edges looks more like a natural
            # disturbance.
            if line_count <= 5:
                return (
                    'NATURAL',
                    f'camera found only {line_count} strong straight edges; '
                    f'evidence is more consistent with an irregular natural fall'
                )

            return (
                'UNCERTAIN',
                f'gap detected but camera found {line_count} strong straight edges; '
                f'cause cannot be confirmed'
            )

        if alert_type == 'ANOMALY':
            return (
                'UNCERTAIN',
                f'height anomaly requires ground confirmation; camera detected '
                f'{line_count} strong straight edges'
            )

        return 'UNCERTAIN', 'camera evidence available but disturbance type is unknown'

    # Mission status (feeds the dashboard)

    def _publish_status(self):
        if self.active and self.current_mission is not None:
            state = 'ENROUTE' if self._goal_handle is not None else 'DISPATCHING'
            status = (
                f'STATE={state} queue={len(self.mission_queue)} '
                f'current={self.current_mission.label} '
                f'outcomes={self.report_seq}/{self.total} '
                f'mode={self.priority_mode}'
            )
        else:
            status = (
                f'STATE=IDLE queue={len(self.mission_queue)} '
                f'outcomes={self.report_seq}/{self.total} '
                f'mode={self.priority_mode}'
            )
        self.status_pub.publish(String(data=status))

    def destroy_node(self):
        self.get_logger().info(
            f'Mission Coordinator shutting down. '
            f'Inspected {self.inspected} sites, {self.report_seq}/{self.total} outcomes.'
        )
        if self._goal_handle is not None:
            self.get_logger().info('Cancelling active Nav2 goal...')
            self._goal_handle.cancel_goal_async()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MissionCoordinator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Ctrl+C may have shut down ROS already, so only clean up if it is
        # still running.
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
