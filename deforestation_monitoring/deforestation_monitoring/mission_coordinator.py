#!/usr/bin/env python3
"""
Mission Coordinator: Receives SuspiciousArea flags and dispatches the
Husky ground rover to inspect each site via Nav2 NavigateToPose.

Supports multiple priority strategies:
  - closest:   nearest to the robot first (from /husky1/odometry)
  - confidence: highest confidence first
  - area:      largest affected area first
  - severity:  highest composite severity first
  - composite: (severity * confidence) / distance — big, certain, nearby first

Refinements over the original:
  - Distance-based modes (closest / composite) are re-ranked on every queue
    tick as the robot moves, so priorities track robot-relative proximity.
  - FIFO tie-break: equal-priority missions dispatch in arrival order.
  - Dedup is time-windowed (dedup_window) as well as distance-based.
  - Rejected/aborted goals are requeued with a priority penalty up to
    max_retries instead of being dropped.
  - Evidence capture uses a persistent latest-image subscription (no nested
    spin_once inside a callback).
  - Nav2 availability is checked non-blocking (no 5 s blocking wait inside
    the queue timer).
  - Publishes /mission_status so the dashboard can show decision state.
  - Active Nav2 goals are cancelled on shutdown.
"""

import cv2
import heapq
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Point, Pose, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import Header, String
from deforestation_interfaces.msg import SuspiciousArea
from cv_bridge import CvBridge, CvBridgeError
from tf2_ros import Buffer, TransformException, TransformListener


@dataclass(order=True)
class Mission:
    """Prioritised inspection task. Lower priority value = higher urgency.

    Comparison uses (priority, seq): equal priorities are broken FIFO by
    seq (lower seq = queued earlier). Defaulted fields stay at the end so
    the dataclass constructor stays positional-compatible.
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
    """Receives SuspiciousArea flags, dispatches Husky via Nav2."""

    def __init__(self):
        super().__init__('mission_coordinator')

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
                description='Nav2 global frame — flags are transformed into it before dispatch'))
        self.declare_parameter('dedup_radius', 10.0,
            descriptor=ParameterDescriptor(
                description='Skip flags within this radius (m) of a queued or '
                            'inspected site — stops repeated detections flooding the queue'))
        self.declare_parameter('dedup_window', 300.0,
            descriptor=ParameterDescriptor(
                description='Time window (s, sim clock) for site dedup — a site '
                            'is only re-queued after this window elapses'))
        self.declare_parameter('max_retries', 3,
            descriptor=ParameterDescriptor(
                description='Max Nav2 goal retries for rejected/aborted missions'))
        self.declare_parameter('requeue_penalty', 50.0,
            descriptor=ParameterDescriptor(
                description='Priority penalty added to a mission on each retry'))

        self.priority_mode = self.get_parameter('priority_mode').value
        self.capture_enabled = self.get_parameter('capture_images').value
        self.image_dir = self.get_parameter('image_dir').value
        self.map_frame = self.get_parameter('map_frame').value
        self.dedup_radius = self.get_parameter('dedup_radius').value
        self.dedup_window = self.get_parameter('dedup_window').value
        self.max_retries = self.get_parameter('max_retries').value
        self.requeue_penalty = self.get_parameter('requeue_penalty').value

        # TF for transforming flag coordinates into the Nav2 map frame
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

        # Recently queued/inspected sites for dedup: (x, y, sim-clock ns)
        self._recent_sites = deque(maxlen=50)

        # Camera capture state — persistent latest-image subscription
        # (replaces the old create-subscribe-then-nested-spin approach)
        self.bridge = CvBridge()
        self._latest_image = None

        # -- QoS profiles --
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

    # ── Priority scoring ─────────────────────────────────────────────

    def _compute_priority_value(self, mode: str, severity: float,
                                confidence: float, area_m2: float,
                                dist: float) -> float:
        """Compute a mission priority value. LOWER value = HIGHER urgency.

        All modes return values where smaller = more urgent.
        """
        if mode == 'closest':
            # Pure distance — closest first
            return dist

        elif mode == 'confidence':
            # Highest confidence first. Negate so higher confidence = lower priority value.
            return -confidence

        elif mode == 'area':
            # Largest area first
            return -area_m2

        elif mode == 'severity':
            # Highest severity first
            return -severity

        elif mode == 'composite':
            # Composite: (severity * confidence) / (distance + 1).
            # Closer, higher-confidence, bigger threats = lower priority value.
            # The +1 prevents division by zero. Negate the score.
            score = (severity * confidence) / (dist + 1.0)
            return -score

        else:
            self.get_logger().warn(
                f'Unknown priority_mode: {mode}, using closest',
                throttle_duration_sec=30.0)
            return dist

    def _distance_to(self, gx: float, gy: float) -> float:
        """Distance from the HUSKY (not the map origin) to a flag.

        Falls back to map origin until Husky odometry arrives.
        """
        if self.robot_x is not None:
            return math.hypot(gx - self.robot_x, gy - self.robot_y)
        self.get_logger().warn(
            'No Husky odometry yet — scoring distance from map origin.',
            throttle_duration_sec=10.0)
        return math.hypot(gx, gy)

    # ── Flag handling ─────────────────────────────────────────────────

    def odom_callback(self, msg: Odometry):
        """Track the Husky pose for robot-relative priority scoring."""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

    def cam_callback(self, msg: Image):
        """Keep the latest camera frame for inspection evidence."""
        self._latest_image = msg

    def flag_callback(self, msg: SuspiciousArea):
        """New suspicious area detected. Queue for inspection."""
        gx = msg.position.x
        gy = msg.position.y
        now_ns = self.get_clock().now().nanoseconds

        # Dedup: repeated detections of the same site would otherwise flood
        # the queue. Skip only if within dedup_radius AND within the
        # dedup_window — old inspections expire, allowing re-inspection.
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

        # Compute priority for this flag using the selected strategy
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
        """Process next mission in the queue if idle."""
        if self.active or not self.mission_queue:
            return

        # Non-blocking Nav2 readiness check (the action server is a
        # lifecycle node — it may still be coming up at launch).
        if not self.nav_client.server_is_ready():
            self.get_logger().warn(
                'Nav2 server not ready — waiting.', throttle_duration_sec=5.0)
            return

        # Distance-based modes: re-rank the whole queue as the robot moves,
        # so proximity reflects the current Husky pose, not the pose at the
        # time each flag was queued.
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

    # ── Navigation ────────────────────────────────────────────────────

    def _requeue(self, mission: Mission, penalty: float):
        """Re-queue a mission with a priority penalty and a retry counter."""
        mission.priority += penalty
        mission.attempts += 1
        heapq.heappush(self.mission_queue, mission)
        self.get_logger().warn(
            f'Requeued {mission.label} (attempt {mission.attempts}/'
            f'{self.max_retries}) with priority penalty {penalty:.0f}. '
            f'Queue: {len(self.mission_queue)}')

    def _send_goal(self, mission: Mission):
        """Send a NavigateToPose goal to Nav2.

        Flags may arrive in a non-Nav2 frame (e.g. parrot1_odom from the
        drone's canopy scan mapper). We transform the flag position into
        the Nav2 map frame via TF when possible; if the TF tree does not
        cover the flag frame, we assume the frames coincide (both are
        anchored at the sim world origin) and use the raw coordinates.
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
                # Ground-plane (yaw-only) rotation of the flag offset
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

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped(
            header=Header(
                stamp=self.get_clock().now().to_msg(),
                frame_id=gframe,
            ),
            pose=Pose(
                position=Point(x=gx, y=gy, z=0.0),
                orientation=Quaternion(w=1.0),
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
        """Handle Nav2 goal completion — capture evidence on success."""
        self._goal_handle = None
        result = future.result()
        status = result.status
        m = self.current_mission

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.inspected += 1
            self.report_seq += 1
            evidence_path = self._capture_evidence()
            evidence_str = f' [image: {evidence_path}]' if evidence_path else ' [no image]'
            classification, reason = self._classify_disturbance(m)

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
                # Transient failure (stuck path, stale costmap): retry with a
                # priority penalty instead of dropping the site.
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

    # ── Evidence capture ──────────────────────────────────────────────

    def _capture_evidence(self) -> str:
        """Save the latest Husky camera frame as inspection evidence.

        Uses the persistent latest-image subscription — no nested spinning
        inside a callback (previously rclpy.spin_once ran inside _result,
        blocking the executor and relying on fragile callback interleaving).

        Returns the file path of the saved image, or empty string on failure.
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

    def _classify_disturbance(self, mission):
        """Stub decision tree: classify a disturbance as NATURAL/CUT/UNCERTAIN.

        MVP heuristic based on the alert type. Phase 2 should analyse the
        captured evidence image (stump, cut log, disturbed soil) plus the
        tree_mapper trunk-loss signal:
          - straight clearing edges / cut trunks -> CUT
          - scattered fallen trees / broken canopy -> NATURAL
          - otherwise -> UNCERTAIN
        """
        t = (mission.type or '').upper()
        if t == 'LINE':
            return 'CUT', 'straight-edge clearing pattern suggests deliberate removal'
        if t == 'GAP':
            return 'UNCERTAIN', 'gap may be natural fall or cutting - ground evidence required'
        if t == 'ANOMALY':
            return 'UNCERTAIN', 'height anomaly needs ground confirmation'
        return 'UNCERTAIN', 'classification stub - image/lidar evidence not yet analysed'

    # ── Mission status (feeds the dashboard) ──────────────────────────

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
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
