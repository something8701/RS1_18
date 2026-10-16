#!/usr/bin/env python3
# The node that runs the ground check with the Husky's lidar.

"""Tree refiner: the Husky's LiDAR checks the drone's trees on the ground.

Inputs: the drone tracker's frozen trees (/parrot_tree_baseline_info) and
LOST events, tree_mapper's per-scan trunk cluster centres
(/tree_cluster_points) and the Husky's scans with TF for clear views.
Logic in tree_refine.py.

Publishes /refined_trees (RefinedTreeArray, latched). ``mode``:
  annotate  the drone's results stay as they are; this is a second layer;
  correct   consumers apply it: a FALSE_LOST cancels the drone's LOST and a
            HIDDEN trunk is a tree. Positions stay the drone's: the Husky's
            trunk positions were no better (oaks) or worse (pines, branches).
"""

import json
import math

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_sensor_data)
from rclpy.time import Time
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import Header, String
from tf2_ros import Buffer, TransformException, TransformListener

from deforestation_interfaces.msg import RefinedTree, RefinedTreeArray, TreeChangeEvent

from .tree_refine import NAMES, RefineConfig, TrunkTracker, clear_views, refine, summary


class TreeRefiner(Node):
    def __init__(self):
        super().__init__('tree_refiner')
        p = lambda n, d: self.declare_parameter(n, d).value
        self.mode = str(p('mode', 'annotate'))


        self.frame = str(p('world_frame', 'husky1_odom'))
        self.cfg = RefineConfig(match_radius=float(p('match_radius', 1.5)),
                                min_hits=int(p('min_hits', 4)),
                                min_views=int(p('min_views', 12)),
                                hidden_clear=float(p('hidden_clear', 2.0)))
        self.max_view_dist = float(p('max_view_dist', 12.0))
        self.trunk_range = float(p('trunk_range', 8.0))
        self.trees = []
        self.lost = {}
        self.views = {}
        self.sensor_xy = None
        self.tracker = TrunkTracker(forget=int(p('forget_views', 8)))
        self._last_summary = None

        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        reliable = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.VOLATILE,
                              history=HistoryPolicy.KEEP_LAST, depth=50)
        self.tf_buffer = Buffer()


        self.tf_listener = TransformListener(self.tf_buffer, None, spin_thread=True)
        self.create_subscription(String, '/parrot_tree_baseline_info', self.info_cb, latched)
        self.create_subscription(TreeChangeEvent, '/parrot_tree_change_events',
                                 self.event_cb, reliable)
        self.create_subscription(PointCloud2, '/tree_cluster_points', self.clusters_cb,
                                 qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/husky1/scan', self.scan_cb,
                                 qos_profile_sensor_data)
        self.pub = self.create_publisher(RefinedTreeArray, '/refined_trees', latched)
        self.create_timer(2.0, self.publish)
        self.get_logger().info(f'Tree refiner ready, mode {self.mode}')

    def info_cb(self, msg: String):
        try:
            info = json.loads(msg.data)
        except ValueError:
            return
        self.trees = [(int(t['id']), float(t['x']), float(t['y']))
                      for t in info.get('trees', []) if t.get('source') != 'camera']

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def event_cb(self, msg: TreeChangeEvent):
        if msg.event_type == 'LOST':
            self.lost[int(msg.tree_id)] = self._now()
            self.views[int(msg.tree_id)] = 0

    def clusters_cb(self, msg: PointCloud2):
        if msg.width == 0 or self.sensor_xy is None:
            return
        a = np.frombuffer(bytes(msg.data), np.float32).reshape(-1, msg.point_step // 4)
        try:
            tf = self.tf_buffer.lookup_transform(self.frame, msg.header.frame_id,
                                                 Time.from_msg(msg.header.stamp),
                                                 timeout=Duration(seconds=0.1))
        except TransformException:
            return
        t, q = tf.transform.translation, tf.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        c, s = math.cos(yaw), math.sin(yaw)
        xs = t.x + c * a[:, 0] - s * a[:, 1]
        ys = t.y + s * a[:, 0] + c * a[:, 1]
        sx, sy = self.sensor_xy
        near = [(float(x), float(y)) for x, y in zip(xs, ys)
                if math.hypot(x - sx, y - sy) <= self.trunk_range]
        self.tracker.add(near, self._now())

    def scan_cb(self, scan: LaserScan):
        try:
            tf = self.tf_buffer.lookup_transform(self.frame, scan.header.frame_id,
                                                 Time.from_msg(scan.header.stamp),
                                                 timeout=Duration(seconds=0.1))
        except TransformException:
            return
        t, q = tf.transform.translation, tf.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.sensor_xy = (t.x, t.y)
        ranges = np.asarray(scan.ranges, np.float32)
        spots = np.array([(x, y) for _, x, y in self.trees])
        seen = clear_views((t.x, t.y), yaw, scan.angle_min, scan.angle_increment,
                           ranges, spots, self.max_view_dist)
        for (tid, _, _), s in zip(self.trees, seen):
            if s:
                self.views[tid] = self.views.get(tid, 0) + 1
        if self.tracker.trunks:
            trunks = np.array([(k.x, k.y) for k in self.tracker.trunks])
            self.tracker.seen_through(clear_views(
                (t.x, t.y), yaw, scan.angle_min, scan.angle_increment, ranges, trunks,
                self.trunk_range))

    def publish(self):
        if not self.trees:
            return
        refined = refine(self.trees, self.lost, self.tracker.trunks, self.views, self.cfg)
        arr = RefinedTreeArray(header=Header(stamp=self.get_clock().now().to_msg(),
                                             frame_id=self.frame), mode=self.mode)
        for r in refined:
            g = r.ground_xy or (0.0, 0.0)
            arr.trees.append(RefinedTree(
                id=r.id, drone_position=Point(x=r.drone_xy[0], y=r.drone_xy[1]),
                ground_position=Point(x=g[0], y=g[1]), status=r.status,
                trunk_hits=min(r.trunk_hits, 65535), views=min(r.views, 65535)))
        self.pub.publish(arr)
        s = summary(refined)
        if s != self._last_summary:
            self._last_summary = s
            self.get_logger().info('ground check: ' + ', '.join(
                f'{s[n]} {n}' for n in NAMES if s[n]))


def main(args=None):
    rclpy.init(args=args)
    node = TreeRefiner()
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
