#!/usr/bin/env python3
# Finds dumped red bags with the Husky's camera and keeps the list.

"""Waste detector: finds dumped waste (red garbage bags) and keeps a list of it.

The Husky's RGB-D camera does the finding. Its coloured point cloud
(``/husky1/camera/depth/points``, x forward in the camera frame) is checked
for red points (waste.red_mask); a group of enough red points between
``min_range`` and ``max_range`` is a sighting, placed on the map through TF.
Two sightings within 1 m make a new site (one placed 2 m off with the bag
filling the view right in front of the camera made false sites), and a
sighting must be about as big as a bag at its distance: a far bag seen
through pine branches gives a few red edge pixels with the branch's depth,
which put it metres from where it is. It runs all the time,
so bags the Husky passes on its way to a suspected area count too, and the
Husky's loop around each suspected area looks at it from every side.

With ``drone_red``, red clusters on the drone camera's ``/forest_red_map``
become sites too ("seen from the air"); the Husky confirms them when it
sees red there.

Publishes ``/waste_sites`` (WasteSiteArray, latched) and ``/waste_markers``;
a photo of each bag is saved to ``photo_dir`` when the Husky first sees it and
replaced when a later view shows clearly more of it.
"""

import math
import os
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_sensor_data)
from rclpy.time import Time
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from deforestation_interfaces.msg import WasteSite, WasteSiteArray

from .waste import BAG_RADII, CONFIRMED, RedBlob, SiteTracker, find_red_blobs, red_mask


def cloud_red_points(msg: PointCloud2, max_range: float, min_range: float = 0.0) -> np.ndarray:
    """(N, 3) camera-frame points of the red pixels of a gz RGB-D cloud
    (fields x, y, z at 0, 4, 8 and packed rgb at 16), ``min_range`` to
    ``max_range`` m from the camera."""
    a = np.frombuffer(bytes(msg.data), np.uint8).reshape(-1, msg.point_step)
    xyz = a[:, :12].copy().view(np.float32).reshape(-1, 3)
    bgr = a[:, 16:19]
    rgb = bgr[:, ::-1]
    dist = np.linalg.norm(xyz, axis=1)
    ok = np.isfinite(xyz).all(1) & (dist <= max_range) & (dist >= min_range)
    ok &= red_mask(rgb)
    return xyz[ok]


def bag_pixels(distance: float, hfov: float = 2.0944, width: int = 720) -> float:
    """Pixels a whole bag covers at ``distance`` m, seen end-on (its smallest
    side, an ellipse of the bag's y and z radii)."""
    f = width / 2 / math.tan(hfov / 2)
    _, ry, rz = BAG_RADII
    return math.pi * ry * rz * (f / max(distance, 0.1)) ** 2


def cluster_xy(points: np.ndarray, cell: float = 0.25, min_points: int = 30):
    """Groups of points on the ground plane: [(x, y, n)] (connected cells)."""
    if len(points) == 0:
        return []
    from scipy import ndimage
    ij = np.floor(points[:, :2] / cell).astype(int)
    lo = ij.min(0)
    ij -= lo
    grid = np.zeros(ij.max(0) + 1, int)
    np.add.at(grid, (ij[:, 0], ij[:, 1]), 1)
    labels, n = ndimage.label(grid > 0, structure=np.ones((3, 3)))
    out = []
    for i in range(1, n + 1):
        sel = labels[ij[:, 0], ij[:, 1]] == i
        if sel.sum() >= min_points:
            c = points[sel].mean(0)
            out.append((float(c[0]), float(c[1]), float(c[2]), int(sel.sum())))
    return out


class WasteDetector(Node):
    def __init__(self):
        super().__init__('waste_detector')
        p = lambda n, d: self.declare_parameter(n, d).value


        self.frame = p('world_frame', 'husky1_odom')
        self.max_range = float(p('max_range', 8.0))
        self.min_range = float(p('min_range', 1.2))
        self.min_fill = float(p('min_fill', 0.2))
        self.min_points = int(p('min_points', 30))
        self.photo_dir = os.path.expanduser(p('photo_dir', '/tmp/deforestation_eval/waste_photos'))
        self.drone_red = bool(p('drone_red', False))
        self.period = float(p('process_period', 0.5))
        self.tracker = SiteTracker(match_dist=float(p('match_dist', 3.0)))
        self.pending = []
        self.photos = {}
        self.photo_points = {}
        self.last_image = None
        self._last_cloud = 0.0
        os.makedirs(self.photo_dir, exist_ok=True)

        self.tf_buffer = Buffer()


        self.tf_listener = TransformListener(self.tf_buffer, None, spin_thread=True)
        self.create_subscription(PointCloud2, p('cloud_topic', '/husky1/camera/depth/points'),
                                 self.cloud_cb, qos_profile_sensor_data)
        self.create_subscription(Image, p('image_topic', '/husky1/camera/image'),
                                 self.image_cb, qos_profile_sensor_data)
        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        if self.drone_red:
            self.create_subscription(OccupancyGrid, '/forest_red_map', self.red_map_cb, latched)
        self.sites_pub = self.create_publisher(WasteSiteArray, '/waste_sites', latched)
        self.marker_pub = self.create_publisher(MarkerArray, '/waste_markers', 10)
        self.create_timer(2.0, self.publish)
        self.get_logger().info(
            f'Waste detector ready: Husky camera up to {self.max_range} m, '
            f'>= {self.min_points} red points per bag; drone red map '
            f'{"on" if self.drone_red else "off"}; photos in {self.photo_dir}')

    def image_cb(self, msg: Image):
        self.last_image = msg

    def cloud_cb(self, msg: PointCloud2):
        now = time.monotonic()
        if now - self._last_cloud < self.period:
            return
        self._last_cloud = now
        red = cloud_red_points(msg, self.max_range, self.min_range)
        if len(red) < self.min_points:
            return
        try:
            tf = self.tf_buffer.lookup_transform(self.frame, msg.header.frame_id,
                                                 Time.from_msg(msg.header.stamp),
                                                 timeout=Duration(seconds=0.2))
        except TransformException as exc:
            self.get_logger().warn(f'no TF for a red sighting: {exc}', throttle_duration_sec=10.0)
            return
        t, q = tf.transform.translation, tf.transform.rotation
        rot = _quat_matrix(q.x, q.y, q.z, q.w)
        for cx, cy, cz, n in cluster_xy(red, min_points=self.min_points):
            if n < self.min_fill * bag_pixels(math.sqrt(cx * cx + cy * cy + cz * cz)):
                continue
            w = rot @ np.array([cx, cy, cz]) + np.array([t.x, t.y, t.z])
            site = self._sighting(float(w[0]), float(w[1]), n)
            if site is None:
                continue
            first = site.id not in self.photos
            if first or n > 1.5 * self.photo_points.get(site.id, 0):
                self.photos[site.id] = self._save_photo(site.id) or self.photos.get(site.id, '')
                self.photo_points[site.id] = n
            if first:
                self.get_logger().info(
                    f'WASTE #{site.id} found by the Husky at ({site.x:.1f}, {site.y:.1f}), '
                    f'{n} red points, photo {self.photos[site.id] or "none"}')

    def _sighting(self, x, y, n):
        site = self.tracker._nearest(x, y)
        if site is None:
            now = time.monotonic()
            self.pending = [q for q in self.pending if now - q[3] < 30.0]
            q = min(self.pending, key=lambda q: math.hypot(q[0] - x, q[1] - y), default=None)
            if q is None or math.hypot(q[0] - x, q[1] - y) > 1.0:
                self.pending.append([x, y, 1, now])
                return None
            self.pending.remove(q)
            x, y = (q[0] + x) / 2, (q[1] + y) / 2
            site = self.tracker.update([RedBlob(x, y, float(n), 0.0)])[0]
        if site.status != CONFIRMED:
            site.status = CONFIRMED
            site.history.append(CONFIRMED)
        return site

    def red_map_cb(self, msg: OccupancyGrid):
        data = np.array(msg.data, np.int16).reshape(msg.info.height, msg.info.width).T
        hits = np.where(data > 0, data, 0).astype(np.float32)
        blobs = find_red_blobs(hits, msg.info.resolution,
                               msg.info.origin.position.x, msg.info.origin.position.y)
        for s in self.tracker.update(blobs):
            self.get_logger().info(f'WASTE #{s.id} seen from the air at ({s.x:.1f}, {s.y:.1f})')

    def _save_photo(self, site_id) -> str:
        if self.last_image is None or self.last_image.encoding not in ('rgb8', 'bgr8'):
            return ''
        try:
            import cv2
            img = np.frombuffer(self.last_image.data, np.uint8).reshape(
                self.last_image.height, self.last_image.width, 3)
            if self.last_image.encoding == 'rgb8':
                img = img[..., ::-1]
            path = os.path.join(self.photo_dir, f'waste_{site_id}.jpg')
            cv2.imwrite(path, img)
            return path
        except Exception as exc:
            self.get_logger().warn(f'photo failed: {exc}')
            return ''

    def publish(self):
        stamp = self.get_clock().now().to_msg()
        arr = WasteSiteArray(header=Header(stamp=stamp, frame_id=self.frame))
        markers = MarkerArray()
        for s in self.tracker.sites:
            arr.sites.append(WasteSite(id=s.id, position=Point(x=s.x, y=s.y, z=0.0),
                                       area_m2=float(s.area_m2), confidence=float(s.confidence),
                                       status=s.status, photo=self.photos.get(s.id, '')))
            m = Marker(header=arr.header, ns='waste', id=s.id, type=Marker.SPHERE,
                       action=Marker.ADD)
            m.pose.position = Point(x=s.x, y=s.y, z=0.3)
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.8
            m.color.r, m.color.g, m.color.b, m.color.a = (
                (1.0, 0.1, 0.1, 0.9) if s.status == CONFIRMED else (1.0, 0.6, 0.0, 0.9))
            markers.markers.append(m)
        self.sites_pub.publish(arr)
        self.marker_pub.publish(markers)


def _quat_matrix(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def main(args=None):
    rclpy.init(args=args)
    node = WasteDetector()
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
