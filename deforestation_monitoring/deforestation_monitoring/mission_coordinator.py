#!/usr/bin/env python3
# mission_coordinator.py: Receives SuspiciousArea flags and dispatches the Husky ground rover to inspect each site via Nav2 …
import cv2
import heapq
import numpy as np
import json
import math
import os
import csv
import time
from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace

import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Point, Pose, Quaternion, Twist, Vector3
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
from std_msgs.msg import Header, String
from deforestation_interfaces.msg import EvidenceArray
from deforestation_interfaces.msg import (MissionInfo, MissionQueue, SuspiciousArea,
                                          TreeChangeEvent, WasteSiteArray)
from cv_bridge import CvBridge, CvBridgeError
from tf2_ros import Buffer, TransformException, TransformListener

from .area_search import box_overlap, free_spot, loss_box, sweep_path


@dataclass(order=True)
class Mission:
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
    site: tuple = field(default=None, compare=False)
    box: tuple = field(default=None, compare=False)
    path: list = field(default_factory=list, compare=False)
    stops: list = field(default_factory=list, compare=False)
    reached: int = field(default=0, compare=False)
    photos: list = field(default_factory=list, compare=False)
    status: int = field(default=MissionInfo.QUEUED, compare=False)
    result: str = field(default='', compare=False)
    staging: tuple = field(default=None, compare=False)
    hops: int = field(default=0, compare=False)
    pinned: bool = field(default=False, compare=False)
    second_try: bool = field(default=False, compare=False)


class MissionCoordinator(Node):

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
                description='Nav2 global frame — flags are transformed into it before dispatch'))
        self.declare_parameter('world_frame', 'husky1_odom',
            descriptor=ParameterDescriptor(
                description='Frame the drone\'s flags are taken in when there is no TF to '
                            'map_frame (the drone and the Husky have separate TF trees). '
                            'The Husky\'s odometry is world-aligned in the sim; its SLAM map '
                            'drifts metres from it, so goals go out in this frame and Nav2 '
                            'moves them into its map'))
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

        self.declare_parameter('inspection_standoff', 1.0,
            descriptor=ParameterDescriptor(
                description='Distance in metres Husky stops from a suspicious area'))
        self.declare_parameter('area_search', False,
            descriptor=ParameterDescriptor(
                description='Sweep each suspected area instead of stopping at the flag: '
                            'the box of canopy loss around it (drone change map) plus '
                            'search_margin, in lanes search_spacing apart, from the corner '
                            'nearest the Husky, a photo at each lane end (looks for dumped '
                            'waste; the LiDAR maps the trunks there)'))
        self.declare_parameter('search_margin', 2.0,
            descriptor=ParameterDescriptor(
                description='How far around the suspected area the sweep goes (m); the '
                            'Husky starts this far outside it'))
        self.declare_parameter('search_spacing', 3.0,
            descriptor=ParameterDescriptor(description='Distance between sweep lanes (m)'))
        self.declare_parameter('waypoint_timeout', 45.0,
            descriptor=ParameterDescriptor(
                description='A sweep goal is given up after this many seconds plus 4 s '
                            'per metre to it (Nav2 recoveries can take minutes)'))
        self.declare_parameter('ask_before_each_area', False,
            descriptor=ParameterDescriptor(
                description='Operator in charge: before each area the Husky waits for '
                            '/mission_command "proceed" (or "choose" another). Also '
                            'switchable at run time with {"cmd": "ask", "on": true}.'))
        self.declare_parameter('wait_for_drone_baseline', False,
            descriptor=ParameterDescriptor(
                description='Keep the Husky still until the drone has frozen its tree '
                            'baseline (/drone_baseline_status "frozen: ..."); flags '
                            'are queued meanwhile'))

        self.area_search = bool(self.get_parameter('area_search').value)
        self.search_margin = float(self.get_parameter('search_margin').value)
        self.search_spacing = float(self.get_parameter('search_spacing').value)
        self.waypoint_timeout = float(self.get_parameter('waypoint_timeout').value)
        self._goal_deadline = None
        self._timed_out = False
        self._sent_from = None
        self._escape_until = None
        self._after_escape = None
        self.robot_yaw = 0.0
        self.hazards = []
        self.wait_for_drone = bool(self.get_parameter('wait_for_drone_baseline').value)
        self.drone_ready = not self.wait_for_drone
        self.priority_mode = self.get_parameter('priority_mode').value
        self.capture_enabled = self.get_parameter('capture_images').value
        self.image_dir = self.get_parameter('image_dir').value
        self.map_frame = self.get_parameter('map_frame').value
        self.world_frame = self.get_parameter('world_frame').value
        self.dedup_radius = self.get_parameter('dedup_radius').value
        self.dedup_window = self.get_parameter('dedup_window').value
        self.max_retries = self.get_parameter('max_retries').value
        self.requeue_penalty = self.get_parameter('requeue_penalty').value
        self.inspection_standoff = self.get_parameter('inspection_standoff').value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        if self.capture_enabled:
            os.makedirs(self.image_dir, exist_ok=True)

        self.mission_queue = []
        self.current_mission = None
        self.done = deque(maxlen=50)
        self.change_map = None
        self.costmap = None
        self.active = False
        self.inspected = 0
        self.report_seq = 0
        self.total = 0
        self._seq_counter = 0
        self._goal_handle = None

        self.robot_x = None
        self.robot_y = None

        self._recent_sites = deque(maxlen=50)

        self.bridge = CvBridge()
        self._latest_image = None

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
        self.queue_pub = self.create_publisher(MissionQueue, '/mission_queue', alert_qos)
        if self.wait_for_drone:
            self.create_subscription(String, '/drone_baseline_status',
                                     self._drone_baseline_cb, alert_qos)
        if self.area_search:
            latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 history=HistoryPolicy.KEEP_LAST, depth=1)
            self.create_subscription(OccupancyGrid, '/canopy_change_map',
                                     lambda m: setattr(self, 'change_map', m), latched)
            self.create_subscription(OccupancyGrid, '/husky1/global_costmap/costmap',
                                     lambda m: setattr(self, 'costmap', m), latched)
            self.waste = []
            self.create_subscription(WasteSiteArray, '/waste_sites', lambda m: setattr(
                self, 'waste', [(w.position.x, w.position.y) for w in m.sites]), latched)
            self.evidence = []
            self.create_subscription(EvidenceArray, '/logging_evidence', lambda m: setattr(
                self, 'evidence', [(e.kind, e.position.x, e.position.y) for e in m.items]), latched)
            self.create_timer(2.0, self._watchdog)
            self.cmd_pub = self.create_publisher(Twist, '/husky1/cmd_vel', 10)
            self.create_timer(0.1, self._escape_tick)
            self.hazard_pub = self.create_publisher(PointCloud2, '/husky1/hazards', 10)
            self.create_timer(1.0, self._publish_hazards)
            self.create_subscription(TreeChangeEvent, '/parrot_tree_change_events',
                                     self._lost_tree_cb, alert_qos)
        self.status_pub = self.create_publisher(
            String, '/mission_status', alert_qos
        )

        self.nav_client = ActionClient(
            self, NavigateToPose, '/husky1/navigate_to_pose'
        )

        self.ask_first = bool(self.get_parameter('ask_before_each_area').value)
        self.paused = False
        self.approved = None
        latched_ctl = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)
        self.control_pub = self.create_publisher(String, '/mission_control', latched_ctl)
        self.ack_pub = self.create_publisher(String, '/mission_command_ack', alert_qos)
        self.create_subscription(String, '/mission_command', self._command_cb, alert_qos)

        self.create_timer(2.0, self.process_queue)
        self.create_timer(2.0, self._publish_status)
        self._publish_status()
        self.get_logger().info(
            f'Mission Coordinator ready. Priority mode: {self.priority_mode}. '
            f'Camera capture: {self.capture_enabled}.'
        )


    def _compute_priority_value(self, mode: str, severity: float,
                                confidence: float, area_m2: float,
                                dist: float) -> float:
        if mode == 'closest':
            return dist

        elif mode == 'confidence':
            return -confidence

        elif mode == 'area':
            return -area_m2

        elif mode == 'severity':
            return -severity

        elif mode == 'composite':
            score = (severity * confidence) / (dist + 1.0)
            return -score

        else:
            self.get_logger().warn(
                f'Unknown priority_mode: {mode}, using closest',
                throttle_duration_sec=30.0)
            return dist

    def _distance_to(self, gx: float, gy: float) -> float:
        if self.robot_x is not None:
            return math.hypot(gx - self.robot_x, gy - self.robot_y)
        self.get_logger().warn(
            'No Husky odometry yet — scoring distance from map origin.',
            throttle_duration_sec=10.0)
        return math.hypot(gx, gy)


    def odom_callback(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.robot_yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    def cam_callback(self, msg: Image):
        self._latest_image = msg

    def flag_callback(self, msg: SuspiciousArea):
        gx = msg.position.x
        gy = msg.position.y
        now_ns = self.get_clock().now().nanoseconds

        for sx, sy, st in self._recent_sites:
            if (now_ns - st) < self.dedup_window * 1e9 and \
                    math.hypot(gx - sx, gy - sy) < self.dedup_radius:
                self.get_logger().info(
                    f'Flag @ ({gx:.1f},{gy:.1f}) within {self.dedup_radius:.0f}m '
                    f'of a recent site — skipping duplicate.',
                    throttle_duration_sec=5.0)
                return

        if self.area_search:
            nb = self._area_box(SimpleNamespace(x=gx, y=gy))
            for m in [*self.done, *([self.current_mission] if self.current_mission else []),
                      *self.mission_queue]:
                b = m.box or self._area_box(m)
                inside = b[0] - self.search_margin <= gx <= b[2] + self.search_margin and \
                    b[1] - self.search_margin <= gy <= b[3] + self.search_margin
                if inside or max(box_overlap(nb, b), box_overlap(b, nb)) >= 0.5:
                    self.get_logger().info(
                        f'Flag @ ({gx:.1f},{gy:.1f}) is in the area of {m.label}: skipped.')
                    return

        dist = self._distance_to(gx, gy)

        self.total += 1
        self._recent_sites.append((gx, gy, now_ns))

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
            label=(msg.description if msg.type == 'TREE_LOST' else
                   f'[{msg.type} #{self.total}] S={msg.severity:.0f} '
                   f'C={msg.confidence:.0f}% {msg.area_m2:.0f}m²'),
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

    def _lost_tree_cb(self, msg: TreeChangeEvent):
        if msg.event_type != 'LOST':
            return
        flag = SuspiciousArea(header=msg.header, type='TREE_LOST', confidence=90.0,
                              area_m2=float(msg.area_m2), severity=50.0,
                              description=f'Lost tree #{msg.tree_id}')
        flag.position.x, flag.position.y = float(msg.x), float(msg.y)
        self.flag_callback(flag)

    def process_queue(self):
        if self.active or not self.mission_queue or self._escape_until is not None:
            return
        if not self.drone_ready:
            self.get_logger().info(
                "Holding the Husky until the drone's tree baseline is frozen.",
                throttle_duration_sec=30.0)
            return

        if not self.nav_client.server_is_ready():
            self.get_logger().warn(
                'Nav2 server not ready — waiting.', throttle_duration_sec=5.0)
            return

        if self.robot_x is not None and \
                self.priority_mode in ('closest', 'composite'):
            re_scored = []
            while self.mission_queue:
                m = heapq.heappop(self.mission_queue)
                dist = math.hypot(m.x - self.robot_x, m.y - self.robot_y)
                m.priority = self._compute_priority_value(
                    self.priority_mode, m.severity, m.confidence,
                    m.area_m2, dist) + (1000.0 if m.second_try else 0.0) - (1e6 if m.pinned else 0.0)
                re_scored.append(m)
            for m in re_scored:
                heapq.heappush(self.mission_queue, m)

        if self.paused:
            self._publish_status()
            return
        if self.ask_first and self.approved is None:
            self._publish_status()
            return
        pick = self.approved or next((m.label for m in self.mission_queue if m.pinned), None)
        mission = self._take(pick) if pick else None
        self.approved = None
        if mission is None:
            mission = heapq.heappop(self.mission_queue)
        mission.pinned = False
        self.current_mission = mission
        self.active = True
        self._publish_status()
        self._send_goal(mission)


    def _take(self, label):
        for i, m in enumerate(self.mission_queue):
            if m.label == label:
                self.mission_queue.pop(i)
                heapq.heapify(self.mission_queue)
                return m
        return None

    def _next_label(self):
        pinned = [m for m in self.mission_queue if m.pinned]
        if pinned:
            return pinned[0].label
        return min(self.mission_queue).label if self.mission_queue else ''

    def _publish_control(self):
        self.control_pub.publish(String(data=json.dumps({
            'ask_first': self.ask_first, 'paused': self.paused,
            'next': self._next_label(), 'approved': self.approved or '',
            'skipped': [m.label for m in self.done if m.status == MissionInfo.CANCELLED],
        })))

    def _command_cb(self, msg: String):
        try:
            c = json.loads(msg.data)
        except ValueError:
            c = {}
        ok, text = self._apply_command(c)
        self.ack_pub.publish(String(data=json.dumps(
            {'id': c.get('id', ''), 'cmd': c.get('cmd', ''), 'ok': ok, 'msg': text})))
        if ok:
            self.get_logger().info(f'Operator: {text}')
        self._publish_status()

    def _apply_command(self, c):
        cmd, label = c.get('cmd', ''), c.get('label', '')
        queued = {m.label: m for m in self.mission_queue}
        if cmd == 'ask':
            self.ask_first = bool(c.get('on'))
            return True, ('The Husky will ask before each area.' if self.ask_first
                          else 'The Husky will choose its next area itself.')
        if cmd == 'pause':
            self.paused = True
            return True, (f'The Husky will stop after {self.current_mission.label}.'
                          if self.active and self.current_mission else 'The Husky is paused.')
        if cmd == 'resume':
            self.paused = False
            return True, 'The Husky carries on.'
        if cmd == 'proceed':
            if self.active:
                return False, f'The Husky is busy with {self.current_mission.label}.'
            label = label or self._next_label()
            if label not in queued:
                return False, f'{label or "That area"} is not waiting to be searched.'
            self.approved, self.paused = label, False
            return True, f'The Husky is going to {label}.'
        if cmd == 'choose':
            if label not in queued:
                return False, f'{label or "That area"} is not waiting to be searched.'
            for m in self.mission_queue:
                m.pinned = m.label == label
            if self.ask_first and not self.active:
                self.approved = label
            return True, f'{label} is next.'
        if cmd == 'unpin':
            for m in self.mission_queue:
                if m.label == label:
                    m.pinned = False
            if self.approved == label and not c.get('keep_approval'):
                self.approved = None
            return True, f'{label} is back in the usual order.'
        if cmd == 'skip':
            m = self._take(label)
            if m is None:
                return False, f'{label or "That area"} is not waiting to be searched.'
            m.status = MissionInfo.CANCELLED
            m.result = 'Skipped by the operator'
            m.pinned = False
            if self.approved == label:
                self.approved = None
            self.done.append(m)
            return True, f'{label} skipped.'
        if cmd == 'unskip':
            for m in list(self.done):
                if m.label == label and m.status == MissionInfo.CANCELLED:
                    self.done.remove(m)
                    m.status, m.result = MissionInfo.QUEUED, ''
                    heapq.heappush(self.mission_queue, m)
                    return True, f'{label} is back in the queue.'
            return False, f'{label or "That area"} was not skipped.'
        return False, f'Unknown command "{cmd}".'


    def _requeue(self, mission: Mission, penalty: float):
        mission.priority += penalty
        mission.attempts += 1
        mission.status = MissionInfo.QUEUED
        heapq.heappush(self.mission_queue, mission)
        self.get_logger().warn(
            f'Requeued {mission.label} (attempt {mission.attempts}/'
            f'{self.max_retries}) with priority penalty {penalty:.0f}. '
            f'Queue: {len(self.mission_queue)}')

    def _send_goal(self, mission: Mission):
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
                    f'No TF {gframe} → {self.map_frame}; taking the flag in '
                    f'{self.world_frame} (both world-aligned)',
                    throttle_duration_sec=60.0)
                gframe = self.world_frame

        mission.site = (gx, gy)
        mission.status = MissionInfo.ACTIVE
        if self.area_search:
            self._start_search(mission, gx, gy)
            return

        inspection_standoff = self.inspection_standoff

        try:
            if gframe == self.world_frame and self.robot_x is not None:
                raise TransformException('odometry is in the goal frame')
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
                stamp=(rclpy.time.Time() if gframe == self.world_frame
                       else self.get_clock().now()).to_msg(),
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
                m.status, m.result = MissionInfo.UNREACHABLE, msg
                self.done.append(m)
            self.current_mission = None
            self.active = False
            self._publish_status()
            return
        self._goal_handle = handle
        self.get_logger().info('Goal accepted. Husky en route.')
        handle.get_result_async().add_done_callback(self._result)

    def _result(self, future):
        self._goal_handle = None
        result = future.result()
        status = result.status
        m = self.current_mission

        if self._timed_out:
            self._timed_out = False
            if status == GoalStatus.STATUS_CANCELED:
                status = GoalStatus.STATUS_ABORTED
        if self.area_search and m is not None and m.path:
            if self._search_step(m, status):
                return
            if m.reached:
                status = GoalStatus.STATUS_SUCCEEDED

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.inspected += 1
            self.report_seq += 1
            evidence_path = self._capture_evidence(m)
            evidence_str = f' [image: {evidence_path}]' if evidence_path else ' [no image]'
            lidar_path = self._save_lidar_evidence(evidence_path)
            classification, reason = self._classify_disturbance(m, evidence_path)

            m.photos.append(evidence_path)
            search_str = (f' [search: {m.reached}/{len(m.path)} waypoints, '
                          f'{sum(1 for p in m.photos if p)} photos]'
                          if self.area_search and m.path else '')
            msg = (
                f'SITE INSPECTED #{self.report_seq}: {m.label} '
                f'@ ({m.x:.1f},{m.y:.1f}) [{classification}]{evidence_str}{search_str}'
            )
            m.status = MissionInfo.INSPECTED
            self.get_logger().info(msg)
            self.get_logger().info(f'  decision: {classification} - {reason}')

        elif status == GoalStatus.STATUS_CANCELED:
            self.report_seq += 1
            msg = f'SITE CANCELLED #{self.report_seq}: {m.label} @ ({m.x:.1f},{m.y:.1f})'
            self.get_logger().warn(msg)
            m.status = MissionInfo.CANCELLED

        elif status == GoalStatus.STATUS_ABORTED:
            if m.attempts < self.max_retries:
                self.get_logger().warn(
                    f'SITE RETRY: {m.label} aborted — requeuing.')
                if self.area_search and self._stuck():
                    self._escape()
                self._requeue(m, self.requeue_penalty)
                self.current_mission = None
                self.active = False
                self._publish_status()
                return
            if self.area_search and not m.second_try:
                self.get_logger().warn(f'{m.label}: no way in for now; trying it again last.')
                m.second_try, m.attempts = True, 0
                m.path, m.stops = [], []
                self._requeue(m, 1000.0)
                m.attempts = 0
                self.current_mission = None
                self.active = False
                self._publish_status()
                return
            self.report_seq += 1
            msg = f'SITE UNREACHABLE #{self.report_seq}: {m.label} @ ({m.x:.1f},{m.y:.1f})'
            self.get_logger().warn(msg)
            m.status = MissionInfo.UNREACHABLE

        else:
            self.report_seq += 1
            msg = f'SITE STATUS={status} #{self.report_seq}: {m.label} @ ({m.x:.1f},{m.y:.1f})'
            self.get_logger().warn(msg)

        self.report_pub.publish(String(data=msg))
        m.result = msg
        self.done.append(m)
        self.current_mission = None
        self.active = False
        self._publish_status()


    def _area_box(self, m, gx=None, gy=None):
        gx = m.x if gx is None else gx
        gy = m.y if gy is None else gy
        cm = self.change_map
        if cm is None:
            return loss_box(np.zeros((1, 1)), 1.0, 1e9, 1e9, gx, gy)
        info = cm.info
        grid = np.asarray(cm.data, np.int8).reshape(info.height, info.width)
        b = loss_box(grid, info.resolution, info.origin.position.x, info.origin.position.y,
                     m.x, m.y)
        dx, dy = gx - m.x, gy - m.y
        return (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy)

    def _start_search(self, m, gx, gy):
        if not m.path:
            rx = self.robot_x if self.robot_x is not None else gx
            ry = self.robot_y if self.robot_y is not None else gy
            m.box = self._area_box(m, gx, gy)
            cost = None
            if self.costmap is not None:
                ci = self.costmap.info
                cost = (np.asarray(self.costmap.data, np.int8).reshape(ci.height, ci.width),
                        ci.resolution, ci.origin.position.x, ci.origin.position.y)
            tf = self._map_tf()
            for x, y, yaw in sweep_path(m.box, (rx, ry), self.search_margin, self.search_spacing):
                x, y = self._off_waste(x, y)
                spot = (x, y)
                if cost:
                    spot = free_spot(*cost, *self._to_map(tf, x, y))
                    spot = self._to_world(tf, *spot) if spot is not None else None
                if spot is not None:
                    m.path.append((spot[0], spot[1], yaw))
            b = m.box
            self.get_logger().info(
                f'Area search of {m.label}: box ({b[0]:.1f},{b[1]:.1f})-({b[2]:.1f},{b[3]:.1f}) '
                f'+ {self.search_margin:.1f} m, {len(m.path)} waypoints')
        if not m.path:
            m.path = [(gx, gy, 0.0)]
        m.stops = list(m.path[1:])
        x, y, yaw = m.path[0]
        self.get_logger().info(f'Dispatching Husky → {m.label}: start of the sweep at ({x:.1f},{y:.1f})')
        self._send_waypoint(m, x, y, yaw)

    def _off_waste(self, x, y, clear=1.5):
        for wx, wy in self.waste:
            d = math.hypot(x - wx, y - wy)
            if d < clear:
                ux, uy = ((x - wx) / d, (y - wy) / d) if d > 1e-3 else (1.0, 0.0)
                x, y = wx + ux * clear, wy + uy * clear
        return x, y

    def _map_tf(self):
        try:
            t = self.tf_buffer.lookup_transform(self.map_frame, self.world_frame, rclpy.time.Time())
        except TransformException:
            return None
        q = t.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return t.transform.translation.x, t.transform.translation.y, math.cos(yaw), math.sin(yaw)

    @staticmethod
    def _to_map(tf, x, y):
        if tf is None:
            return x, y
        tx, ty, c, s = tf
        return tx + c * x - s * y, ty + s * x + c * y

    @staticmethod
    def _to_world(tf, x, y):
        if tf is None:
            return x, y
        tx, ty, c, s = tf
        dx, dy = x - tx, y - ty
        return c * dx + s * dy, -s * dx + c * dy

    def _staging(self, x, y):
        cm = self.costmap
        if cm is None or self.robot_x is None:
            return None
        tf = self._map_tf()
        x, y = self._to_map(tf, x, y)
        i, m = cm.info, 3.0
        x0, y0 = i.origin.position.x + m, i.origin.position.y + m
        x1 = i.origin.position.x + i.width * i.resolution - m
        y1 = i.origin.position.y + i.height * i.resolution - m
        if x0 <= x <= x1 and y0 <= y <= y1:
            return None
        rx, ry = self._to_map(tf, self.robot_x, self.robot_y)
        for k in range(1, 101):
            px, py = x + (rx - x) * k / 100, y + (ry - y) * k / 100
            if x0 <= px <= x1 and y0 <= py <= y1:
                return self._to_world(tf, px, py) if math.hypot(px - rx, py - ry) > 2.0 else None
        return None

    def _free_now(self, x, y):
        if self.costmap is None:
            return x, y
        ci, tf = self.costmap.info, self._map_tf()
        cost = np.asarray(self.costmap.data, np.int8).reshape(ci.height, ci.width)
        spot = free_spot(cost, ci.resolution, ci.origin.position.x, ci.origin.position.y,
                         *self._to_map(tf, x, y))
        return self._to_world(tf, *spot) if spot is not None else (x, y)

    def _send_waypoint(self, m, x, y, yaw):
        x, y = self._free_now(x, y)
        stage = self._staging(x, y) if m.hops < 4 else None
        if stage is not None:
            stage = self._free_now(*stage)
            m.staging, m.hops = (x, y, yaw), m.hops + 1
            self.get_logger().info(
                f'({x:.1f},{y:.1f}) is beyond the Husky\'s map so far: '
                f'driving to ({stage[0]:.1f},{stage[1]:.1f}) first')
            x, y, yaw = stage[0], stage[1], math.atan2(y - stage[1], x - stage[0])
        else:
            m.staging, m.hops = None, 0
        d = math.hypot(x - self.robot_x, y - self.robot_y) if self.robot_x is not None else 20.0
        self._goal_deadline = time.monotonic() + self.waypoint_timeout + 4.0 * d
        self._goal_xy = (x, y)
        self._sent_from = (self.robot_x, self.robot_y) if self.robot_x is not None else None
        self._send_pose(x, y, yaw)

    def _stuck(self) -> bool:
        f = self._sent_from
        return f is not None and self.robot_x is not None and \
            math.hypot(self.robot_x - f[0], self.robot_y - f[1]) < 0.5

    def _escape(self, then=None):
        hx = self.robot_x + 0.9 * math.cos(self.robot_yaw)
        hy = self.robot_y + 0.9 * math.sin(self.robot_yaw)
        self.hazards.append((hx, hy))
        self.get_logger().warn(
            f'The Husky is stuck: backing out; hazard marked at ({hx:.1f},{hy:.1f}).')
        self._escape_until, self._after_escape, self._sent_from = time.monotonic() + 4.0, then, None

    def _publish_hazards(self):
        if not self.hazards:
            return
        pts = [(hx + dx, hy + dy, 0.3) for hx, hy in self.hazards
               for dx in np.arange(-0.5, 0.51, 0.1) for dy in np.arange(-0.5, 0.51, 0.1)
               if dx * dx + dy * dy <= 0.25]
        hdr = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.world_frame)
        self.hazard_pub.publish(create_cloud_xyz32(hdr, pts))

    def _escape_tick(self):
        if self._escape_until is None:
            return
        if time.monotonic() < self._escape_until:
            self.cmd_pub.publish(Twist(linear=Vector3(x=-0.3)))
            return
        self.cmd_pub.publish(Twist())
        self._escape_until = None
        then, self._after_escape = self._after_escape, None
        if then is not None:
            then()

    def _watchdog(self):
        if self._goal_deadline is None or self._goal_handle is None:
            return
        if time.monotonic() > self._goal_deadline:
            self._goal_deadline = None
            self._timed_out = True
            self.get_logger().warn('Sweep goal is taking too long: giving it up.')
            self._goal_handle.cancel_goal_async()

    def _search_step(self, m, status) -> bool:
        ok = status == GoalStatus.STATUS_SUCCEEDED
        self._goal_deadline = None
        gx, gy = getattr(self, '_goal_xy', (None, None))
        if not ok and status != GoalStatus.STATUS_CANCELED and gx is not None and \
                self.robot_x is not None and math.hypot(gx - self.robot_x, gy - self.robot_y) <= 2.5:
            ok = True
        if m.staging is not None:
            x, y, yaw = m.staging
            if ok:
                self._send_waypoint(m, x, y, yaw)
                return True
            m.staging = None
        if m.reached == 0 and not ok:
            if len(m.path) > 2:
                m.path.pop(0)
            return False
        if ok:
            m.reached += 1
            m.photos.append(self._capture_evidence(m))
        elif status == GoalStatus.STATUS_CANCELED:
            m.stops = []
        if not m.stops:
            return False
        x, y, yaw = m.stops.pop(0)
        self.get_logger().info(
            f'Searching {m.label}: waypoint {len(m.path) - len(m.stops)}/{len(m.path)} '
            f'at ({x:.1f},{y:.1f})')
        if not ok and self._stuck():
            self._escape(lambda: self._send_waypoint(m, x, y, yaw))
        else:
            self._send_waypoint(m, x, y, yaw)
        self._publish_status()
        return True

    def _send_pose(self, x, y, yaw):
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped(
            header=Header(stamp=rclpy.time.Time().to_msg(), frame_id=self.world_frame),
            pose=Pose(position=Point(x=x, y=y, z=0.0),
                      orientation=Quaternion(z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))))
        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(self._goal_response)


    def _capture_evidence(self, mission=None) -> str:
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
        if mission is not None:
            filename = f'site{mission.seq + 1:02d}_{len(mission.photos) + 1:02d}_{ts}.jpg'
        else:
            filename = f'inspection_{self.inspected:03d}_{ts}.jpg'
        filepath = os.path.join(self.image_dir, filename)
        cv2.imwrite(filepath, frame)
        self.get_logger().info(f'Evidence saved: {filepath}')
        return filepath

    def _lidar_callback(self, msg):
        self.latest_scan = msg

    def _save_lidar_evidence(self, image_path=''):
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
        felled = self._felled_evidence(mission)
        if felled:
            st = sum(1 for k in felled if k == 'stump')
            lg = len(felled) - st
            seen = ', '.join(t for t in (f'{st} stump{"s" * (st != 1)}' if st else '',
                                         f'{lg} felled log{"s" * (lg != 1)}' if lg else '') if t)
            return 'CUT', f'the Husky camera (YOLO) saw {seen} in the area'

        if not evidence_path or not os.path.exists(evidence_path):
            return 'UNCERTAIN', 'no usable camera evidence available'

        frame = cv2.imread(evidence_path)
        if frame is None:
            return 'UNCERTAIN', 'camera evidence could not be loaded'

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

    def _felled_evidence(self, mission):
        ev = [e for e in getattr(self, 'evidence', []) if e[0] in ('stump', 'log')]
        box = mission.box
        if box:
            return [k for k, x, y in ev
                    if box[0] - 2.0 <= x <= box[2] + 2.0 and box[1] - 2.0 <= y <= box[3] + 2.0]
        return [k for k, x, y in ev if math.hypot(x - mission.x, y - mission.y) <= 4.0]


    def _drone_baseline_cb(self, msg):
        if not self.drone_ready and msg.data.startswith('frozen'):
            self.drone_ready = True
            self.get_logger().info(
                f'Drone baseline frozen ({msg.data}): the Husky may go out now.')
            self._publish_status()

    def _info(self, m) -> MissionInfo:
        if m is None:
            return MissionInfo()
        box = m.box or (self._area_box(m) if self.area_search else (m.x, m.y, m.x, m.y))
        return MissionInfo(
            label=m.label, type=m.type, position=Point(x=m.x, y=m.y, z=0.0),
            area_min=Point(x=float(box[0]), y=float(box[1])),
            area_max=Point(x=float(box[2]), y=float(box[3])),
            priority=float(m.priority), attempts=min(255, int(m.attempts)), status=m.status,
            path=[Point(x=float(x), y=float(y), z=float(yaw)) for x, y, yaw in m.path],
            reached=min(65535, m.reached), photos=[p for p in m.photos if p], result=m.result)

    def _publish_queue(self, state):
        msg = MissionQueue(state=state, drone_ready=self.drone_ready,
                           current=self._info(self.current_mission),
                           queue=[self._info(q) for q in sorted(self.mission_queue)],
                           done=[self._info(d) for d in self.done],
                           inspected=self.inspected, total=self.total)
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.world_frame
        self.queue_pub.publish(msg)

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
        if not self.drone_ready:
            state = 'WAITING_FOR_DRONE'
        elif self.active and self.current_mission is not None:
            m = self.current_mission
            state = ('SEARCHING' if self.area_search and m.reached > 0
                     else 'ENROUTE' if self._goal_handle is not None else 'DISPATCHING')
        elif self.paused and self.mission_queue:
            state = 'PAUSED'
        elif self.ask_first and self.mission_queue:
            state = 'AWAITING_APPROVAL'
        else:
            state = 'IDLE'
        self._publish_queue(state)
        self._publish_control()

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
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
