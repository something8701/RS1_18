#!/usr/bin/env python3
# yolo_detector.py: Finds tree stumps, felled logs and waste bags in the Husky camera
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_sensor_data)
from rclpy.time import Time
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from deforestation_interfaces.msg import Evidence, EvidenceArray

KINDS = ['stump', 'log', 'bag']
BOX_BGR = {'stump': (0, 200, 255), 'log': (255, 140, 0), 'bag': (40, 40, 255)}
MARK_RGB = {'stump': (1.0, 0.75, 0.1), 'log': (0.2, 0.55, 1.0), 'bag': (1.0, 0.15, 0.15)}
MATCH = {'stump': 1.2, 'log': 2.5, 'bag': 1.2}


@dataclass
class Item:
    id: int
    kind: str
    x: float
    y: float
    conf: float
    n: int = 1
    weight: float = 0.0
    photo: str = ''
    photo_area: float = 0.0
    published: bool = False

    def add(self, x: float, y: float, conf: float):
        w = conf
        tot = self.weight + w
        self.x = (self.x * self.weight + x * w) / tot
        self.y = (self.y * self.weight + y * w) / tot
        self.weight = tot
        self.conf = max(self.conf, conf)
        self.n += 1


class EvidenceTracker:

    def __init__(self):
        self.items = []

    def add(self, kind: str, x: float, y: float, conf: float) -> Item:
        best = min((i for i in self.items if i.kind == kind),
                   key=lambda i: math.hypot(i.x - x, i.y - y), default=None)
        if best is not None and math.hypot(best.x - x, best.y - y) <= MATCH[kind]:
            best.add(x, y, conf)
            return best
        it = Item(id=len(self.items) + 1, kind=kind, x=x, y=y, conf=conf, weight=conf)
        self.items.append(it)
        return it


def box_distance(depth: np.ndarray, x0: int, y0: int, x1: int, y1: int,
                 near: float, far: float, pct: float = 25.0, exclude=(),
                 min_valid: float = 0.5) -> float:
    w, h = x1 - x0, y1 - y0
    ix0, ix1 = max(int(x0 + 0.2 * w), 0), int(math.ceil(x1 - 0.2 * w))
    iy0, iy1 = max(int(y0 + 0.2 * h), 0), int(math.ceil(y1 - 0.2 * h))
    ix1, iy1 = max(ix1, ix0 + 1), max(iy1, iy0 + 1)
    patch = depth[iy0:iy1, ix0:ix1]
    mine = np.ones(patch.shape, bool)
    for ex0, ey0, ex1, ey1 in exclude:
        mine[max(int(ey0) - iy0, 0):max(int(math.ceil(ey1)) - iy0, 0),
             max(int(ex0) - ix0, 0):max(int(math.ceil(ex1)) - ix0, 0)] = False
    keep = mine & np.isfinite(patch) & (patch > near) & (patch < far)
    d = patch[keep]
    if d.size < 3 or d.size < min_valid * mine.sum():
        return float('nan')
    return float(np.percentile(d, pct))


def pixel_ray(u: float, v: float, width: int, height: int, hfov: float):
    f = (width / 2) / math.tan(hfov / 2)
    return 1.0, -(u - width / 2) / f, -(v - height / 2) / f


def quat_matrix(x, y, z, w) -> np.ndarray:
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        p = lambda n, d: self.declare_parameter(n, d).value
        self.frame = p('world_frame', 'husky1_odom')
        weights = os.path.expanduser(p('weights', '~/rs1_18_ws/eval/yolo/weights/evidence_v1.pt'))
        self.conf_min = float(p('conf', 0.5))
        self.min_sightings = int(p('min_sightings', 2))
        self.near = float(p('min_range', 0.6))
        self.far = float(p('max_range', 9.5))
        self.hfov = float(p('hfov', 2.0944))
        self.period = float(p('process_period', 0.25))
        self.photo_dir = os.path.expanduser(p('photo_dir', '/tmp/deforestation_eval/evidence_photos'))
        device = p('device', '0')
        os.makedirs(self.photo_dir, exist_ok=True)

        os.environ['YOLO_AUTOINSTALL'] = 'false'
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.device = device
        self.model.predict(np.zeros((480, 720, 3), np.uint8), device=device, verbose=False)
        self.tracker = EvidenceTracker()
        self.depth = {}
        self._last = 0.0
        self.frames = 0
        self.infer_ms = 0.0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, None, spin_thread=True)
        self.create_subscription(Image, p('image_topic', '/husky1/camera/image'),
                                 self.image_cb, qos_profile_sensor_data)
        self.create_subscription(Image, p('depth_topic', '/husky1/camera/depth/image'),
                                 self.depth_cb, qos_profile_sensor_data)
        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.items_pub = self.create_publisher(EvidenceArray, '/logging_evidence', latched)
        self.marker_pub = self.create_publisher(MarkerArray, '/evidence_markers', 10)
        self.view_pub = self.create_publisher(CompressedImage, '/husky1/yolo/compressed', 10)
        self.create_timer(1.0, self.publish)
        self.get_logger().info(f'YOLO detector ready: {weights} on device {device}, '
                               f'conf >= {self.conf_min}, {self.near}-{self.far} m')

    @staticmethod
    def _key(msg) -> int:
        return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

    def depth_cb(self, msg: Image):
        self.depth[self._key(msg)] = msg
        while len(self.depth) > 6:
            self.depth.pop(min(self.depth))

    def image_cb(self, msg: Image):
        now = time.monotonic()
        if now - self._last < self.period or msg.encoding not in ('rgb8', 'bgr8'):
            return
        self._last = now
        import cv2
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        bgr = img[..., ::-1].copy() if msg.encoding == 'rgb8' else img.copy()
        t0 = time.monotonic()
        res = self.model.predict(bgr, conf=self.conf_min, device=self.device, verbose=False)[0]
        self.infer_ms = 0.9 * self.infer_ms + 0.1 * 1000 * (time.monotonic() - t0) \
            if self.frames else 1000 * (time.monotonic() - t0)
        self.frames += 1
        dmsg = self.depth.get(self._key(msg))
        depth = (np.frombuffer(dmsg.data, np.float32).reshape(dmsg.height, dmsg.width)
                 if dmsg is not None and dmsg.encoding == '32FC1' else None)
        tf = None
        if len(res.boxes) and depth is not None:
            try:
                tf = self.tf_buffer.lookup_transform(self.frame, dmsg.header.frame_id,
                                                     Time.from_msg(msg.header.stamp),
                                                     timeout=Duration(seconds=0.2))
            except TransformException as exc:
                self.get_logger().warn(f'no TF for a detection: {exc}', throttle_duration_sec=10.0)
        if tf is not None:
            t, q = tf.transform.translation, tf.transform.rotation
            rot, trans = quat_matrix(q.x, q.y, q.z, q.w), np.array([t.x, t.y, t.z])
        new_items, better = [], {}
        boxes = res.boxes.xyxy.cpu().numpy()
        for j, (xyxy, c, k) in enumerate(zip(boxes, res.boxes.conf.cpu().numpy(),
                                             res.boxes.cls.cpu().numpy().astype(int))):
            kind = KINDS[k]
            x0, y0, x1, y1 = (float(v) for v in xyxy)
            label = f'{kind} {c:.2f}'
            if tf is not None:
                d = box_distance(depth, int(x0), int(y0), int(x1), int(y1), self.near, self.far,
                                 exclude=[b for i, b in enumerate(boxes) if i != j])
                if math.isfinite(d):
                    ray = np.array(pixel_ray((x0 + x1) / 2, (y0 + y1) / 2, msg.width,
                                             msg.height, self.hfov)) * d
                    w = rot @ ray + trans
                    it = self.tracker.add(kind, float(w[0]), float(w[1]), float(c))
                    label = f'{kind} #{it.id} {c:.2f} {d:.1f}m'
                    if it.n >= self.min_sightings and not it.published:
                        it.published = True
                        new_items.append(it)
                    area = (x1 - x0) * (y1 - y0)
                    if it.published and area > 1.5 * it.photo_area:
                        it.photo_area = area
                        better[it.id] = it
            col = BOX_BGR[kind]
            cv2.rectangle(bgr, (int(x0), int(y0)), (int(x1), int(y1)), col, 2)
            cv2.putText(bgr, label, (int(x0), max(int(y0) - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, col, 1, cv2.LINE_AA)
        cv2.putText(bgr, f'YOLO {self.infer_ms:.0f} ms', (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)
        for it in better.values():
            it.photo = os.path.join(self.photo_dir, f'{it.kind}_{it.id}.jpg')
            cv2.imwrite(it.photo, bgr)
        for it in new_items:
            self.get_logger().info(f'EVIDENCE {it.kind} #{it.id} at ({it.x:.1f}, {it.y:.1f}), '
                                   f'score {it.conf:.2f}, photo {it.photo}')
        ok, jpg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            self.view_pub.publish(CompressedImage(header=msg.header, format='jpeg',
                                                  data=jpg.tobytes()))

    def publish(self):
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.frame)
        arr = EvidenceArray(header=header)
        markers = MarkerArray()
        for it in self.tracker.items:
            if not it.published:
                continue
            arr.items.append(Evidence(id=it.id, kind=it.kind, position=Point(x=it.x, y=it.y),
                                      confidence=float(it.conf), sightings=it.n, photo=it.photo))
            m = Marker(header=header, ns=it.kind, id=it.id, action=Marker.ADD,
                       type=Marker.CYLINDER if it.kind != 'bag' else Marker.SPHERE)
            m.pose.position = Point(x=it.x, y=it.y, z=0.3)
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = 0.7
            m.scale.z = 0.6
            m.color.r, m.color.g, m.color.b = MARK_RGB[it.kind]
            m.color.a = 0.9
            markers.markers.append(m)
        self.items_pub.publish(arr)
        self.marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetector()
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
