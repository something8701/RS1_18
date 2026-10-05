#!/usr/bin/env python3
"""Camera species mapper: what the drone camera sees, on the map grid.

Subscribes to the Parrot RGB and depth images (same sensor, same stamps) and
its odometry, places every pixel on the scan_mapper grid (camera_species)
and publishes three OccupancyGrids, -1 = no data:

  /forest_species_map      mean blue/green x100 of the canopy pixels so far.
                           Oak treetops read ~39, pines ~70; the tracker uses
                           it to find pine tips the height map misses.
  /forest_canopy_seen_map  canopy fraction x100 so far: of the camera rays
                           crossing `open_height` over a cell, the share that
                           hit canopy there. The tracker stores each tree's
                           value at the baseline freeze.
  /forest_canopy_seen      the same fraction from the latest frame that saw
                           the cell, for cells seen in the last `fresh_s`
                           seconds. After a cut the camera sees through the
                           spot (dense pine_111: 0.89 -> 0.00) even where the
                           LiDAR's highest reading is held up by a neighbour's
                           branch tips (4.7 -> 4.2 m).
"""

from collections import deque

import numpy as np
import rclpy
from message_filters import Subscriber, TimeSynchronizer
from nav_msgs.msg import MapMetaData, OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_sensor_data)
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from .camera_species import cell_colour, cell_counts, open_rays, project, quat_to_matrix


def _stamp(msg) -> float:
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


class CameraSpeciesMapper(Node):
    def __init__(self):
        super().__init__('camera_species_mapper')

        def param(name, default, text):
            self.declare_parameter(name, default,
                                   ParameterDescriptor(description=text))
            return self.get_parameter(name).value

        self.altitude = float(param('altitude', 10.0, 'Survey altitude (m); odometry z is 0'))
        self.res = float(param('resolution', 0.25, 'Grid cell size (m), as scan_mapper'))
        self.origin = float(param('origin', -40.0, 'Grid origin x = y (m), as scan_mapper'))
        self.dim = int(param('dim', 320, 'Grid cells per side, as scan_mapper'))
        # Short, so that each lane pass over a spot is its own camera visit
        # for the tracker (tools/eval/camera_visits.py: visits split at
        # ~15 s give the see-through evidence 2-3x sooner than loop-long ones).
        self.fresh_s = float(param('fresh_s', 3.0, 'Recent map keeps cells seen this recently (s)'))
        self.open_height = float(param('open_height', 3.0,
                                       'Canopy fraction is measured at this height (m)'))
        self.frame = str(param('map_frame', 'parrot1_odom', 'Frame of the published grids'))
        image_topic = param('image_topic', '/parrot1/camera/image', 'RGB image')
        depth_topic = param('depth_topic', '/parrot1/camera/depth/image', 'Depth image (32FC1)')

        shape = (self.dim, self.dim)
        self.colour_count = np.zeros(shape, np.float32)
        self.colour_sum = np.zeros(shape, np.float32)
        self.high = np.zeros(shape, np.float32)
        self.open = np.zeros(shape, np.float32)
        self.seen_recent = np.full(shape, np.nan, np.float32)
        self.seen_t = np.full(shape, -np.inf)
        self.odom = deque(maxlen=200)
        self.frames = 0

        self.create_subscription(Odometry, '/parrot1/odometry', self.odom_cb, 50)
        rgb = Subscriber(self, Image, image_topic, qos_profile=qos_profile_sensor_data)
        depth = Subscriber(self, Image, depth_topic, qos_profile=qos_profile_sensor_data)
        self.sync = TimeSynchronizer([rgb, depth], 10)
        self.sync.registerCallback(self.frame_cb)
        # Same QoS as scan_mapper's maps: the tracker subscribes to every map
        # transient-local, and a volatile publisher is never matched to it.
        map_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.species_pub = self.create_publisher(OccupancyGrid, '/forest_species_map', map_qos)
        self.seen_map_pub = self.create_publisher(OccupancyGrid, '/forest_canopy_seen_map', map_qos)
        self.seen_pub = self.create_publisher(OccupancyGrid, '/forest_canopy_seen', map_qos)
        self.create_timer(1.0, self.publish)
        self.get_logger().info(
            f'Camera species mapper ready: {self.dim}x{self.dim} at {self.res} m, '
            f'altitude {self.altitude} m, canopy fraction at {self.open_height} m, '
            f'recent window {self.fresh_s} s')

    def odom_cb(self, msg: Odometry):
        self.odom.append((_stamp(msg), msg))

    def _pose_at(self, t):
        if not self.odom:
            return None
        best = min(self.odom, key=lambda o: abs(o[0] - t))
        return best[1] if abs(best[0] - t) <= 0.1 else None

    def frame_cb(self, rgb_msg: Image, depth_msg: Image):
        t = _stamp(rgb_msg)
        odom = self._pose_at(t)
        if odom is None or rgb_msg.encoding != 'rgb8' or depth_msg.encoding != '32FC1':
            return
        rgb = np.frombuffer(rgb_msg.data, np.uint8).reshape(rgb_msg.height, rgb_msg.width, 3)
        depth = np.frombuffer(depth_msg.data, np.float32).reshape(
            depth_msg.height, depth_msg.width)
        q = odom.pose.pose.orientation
        rot = quat_to_matrix(q.x, q.y, q.z, q.w)
        x, y = odom.pose.pose.position.x, odom.pose.pose.position.y
        grid = (self.res, self.origin, self.origin, self.dim, self.dim)

        points, colours = project(rgb, depth, rot, x, y, self.altitude)
        cells, count, bg_sum = cell_colour(points, colours, *grid)
        self.colour_count.flat[cells] += count
        self.colour_sum.flat[cells] += bg_sum

        tall = points[:, 2] >= self.open_height
        h_cells, h = cell_counts(points[tall, :2], *grid)
        o_cells, o = cell_counts(open_rays(depth, rot, x, y, self.altitude, self.open_height),
                                 *grid)
        self.high.flat[h_cells] += h
        self.open.flat[o_cells] += o
        # This frame's fraction for every cell it saw at open_height.
        u, inv = np.unique(np.concatenate([h_cells, o_cells]), return_inverse=True)
        fh = np.bincount(inv, np.concatenate([h, np.zeros_like(o)]))
        fo = np.bincount(inv, np.concatenate([np.zeros_like(h), o]))
        self.seen_recent.flat[u] = fh / (fh + fo)
        self.seen_t.flat[u] = t
        self.frames += 1

    def _grid(self, values, stamp) -> OccupancyGrid:
        data = np.where(np.isfinite(values), np.clip(np.round(values * 100), 0, 100), -1)
        info = MapMetaData(resolution=self.res, width=self.dim, height=self.dim)
        info.origin.position.x = self.origin
        info.origin.position.y = self.origin
        info.origin.orientation.w = 1.0
        # OccupancyGrid is row-major [iy, ix]; the arrays here are [ix, iy].
        return OccupancyGrid(header=Header(stamp=stamp, frame_id=self.frame), info=info,
                             data=data.T.astype(np.int8).ravel().tolist())

    def publish(self):
        if self.frames == 0:
            return
        stamp = self.get_clock().now().to_msg()
        now = stamp.sec + stamp.nanosec * 1e-9
        colour = np.where(self.colour_count > 0,
                          self.colour_sum / np.maximum(self.colour_count, 1), np.nan)
        seen = self.high + self.open
        seen_all = np.where(seen > 0, self.high / np.maximum(seen, 1), np.nan)
        recent = np.where(now - self.seen_t <= self.fresh_s, self.seen_recent, np.nan)
        self.species_pub.publish(self._grid(colour, stamp))
        self.seen_map_pub.publish(self._grid(seen_all, stamp))
        self.seen_pub.publish(self._grid(recent, stamp))


def main(args=None):
    rclpy.init(args=args)
    node = CameraSpeciesMapper()
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
