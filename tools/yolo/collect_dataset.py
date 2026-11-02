#!/usr/bin/env python3
# collect_dataset.py: Collect an auto-labelled YOLO dataset of stumps, logs and bags in Gazebo
import argparse
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image

from deforestation_monitoring.logging_evidence import (
    CLASSES, evidence_sdf, footprint_radius, model_z, place_ok, random_params, surface_points)
from deforestation_monitoring.tree_detection import parse_tree_truth

W, H, HFOV = 720, 480, 2.0944
FX = (W / 2) / math.tan(HFOV / 2)
CAM_Z = 0.382


def datacam_sdf(rate: float) -> str:
    return f'''<sdf version="1.8"><model name="datacam"><static>true</static>
<pose>0 0 -30 0 0 0</pose><link name="link"><sensor name="cam" type="rgbd_camera">
<camera><horizontal_fov>{HFOV}</horizontal_fov><image><width>{W}</width><height>{H}</height>
</image><clip><near>0.1</near><far>60</far></clip>
<depth_camera><clip><near>0.1</near><far>60</far></clip></depth_camera></camera>
<always_on>1</always_on><update_rate>{rate}</update_rate><topic>/datacam</topic>
</sensor></link></model></sdf>'''


def ign_call(service: str, reqtype: str, req: str, timeout_s: float = 10.0) -> bool:
    for _ in range(3):
        try:
            r = subprocess.run(['ign', 'service', '-s', service, '--reqtype', reqtype,
                                '--reptype', 'ignition.msgs.Boolean', '--timeout', '6000',
                                '--req', req], capture_output=True, text=True, timeout=timeout_s)
            if r.returncode == 0 and 'data: true' in r.stdout:
                return True
        except subprocess.TimeoutExpired:
            pass
        time.sleep(0.3)
    FAILED[service.rsplit('/', 1)[-1]] = FAILED.get(service.rsplit('/', 1)[-1], 0) + 1
    return False


FAILED = {}


def quat(roll: float, pitch: float, yaw: float):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


def pose_fields(name, x, y, z, roll=0.0, pitch=0.0, yaw=0.0) -> str:
    qx, qy, qz, qw = quat(roll, pitch, yaw)
    return (f'name: "{name}" position {{x: {x:.4f} y: {y:.4f} z: {z:.4f}}} '
            f'orientation {{x: {qx:.6f} y: {qy:.6f} z: {qz:.6f} w: {qw:.6f}}}')


def pose_req(name, x, y, z, roll=0.0, pitch=0.0, yaw=0.0) -> str:
    return f'pose {{{pose_fields(name, x, y, z, roll, pitch, yaw)}}} '


def rot_matrix(roll, pitch, yaw) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch),
                              math.sin(pitch), math.cos(yaw), math.sin(yaw))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def label_objects(depth: np.ndarray, cam, objects, min_px: int = 6):
    R = rot_matrix(*cam[3:])
    out = []
    for ob in objects:
        pc = (ob['pts'] - np.array(cam[:3])) @ R
        front = pc[:, 0] > 0.15
        if front.sum() < 4:
            continue
        pc = pc[front]
        u = W / 2 - FX * pc[:, 1] / pc[:, 0]
        v = H / 2 - FX * pc[:, 2] / pc[:, 0]
        inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if inside.sum() < 4:
            continue
        u, v, d = u[inside], v[inside], pc[inside, 0]
        dimg = depth[v.astype(int), u.astype(int)]
        tol = 0.10 + 0.03 * d
        vis = np.isfinite(dimg) & (np.abs(dimg - d) < tol)
        if vis.sum() < 5:
            continue
        uu, vv = u[vis], v[vis]
        x0, x1 = max(uu.min() - 2, 0), min(uu.max() + 2, W - 1)
        y0, y1 = max(vv.min() - 2, 0), min(vv.max() + 2, H - 1)
        if x1 - x0 < min_px or y1 - y0 < min_px:
            continue
        out.append((ob['cls'], (x0 + x1) / 2 / W, (y0 + y1) / 2 / H,
                    (x1 - x0) / W, (y1 - y0) / H, float(vis.mean()), float(d.min())))
    return out


class Collector(Node):
    def __init__(self):
        super().__init__('yolo_dataset_collector')
        self.rgb = {}
        self.depth = {}
        self.n_rgb = 0
        self.sim_ns = 0
        self.create_subscription(Clock, '/clock', self._on_clock, 10)
        self.create_subscription(Image, '/datacam/image', self._rgb, qos_profile_sensor_data)
        self.create_subscription(Image, '/datacam/depth_image', self._depth, qos_profile_sensor_data)

    def _key(self, m):
        return m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec

    def _on_clock(self, m):
        self.sim_ns = m.clock.sec * 1_000_000_000 + m.clock.nanosec

    def sim_now(self) -> int:
        while self.sim_ns == 0:
            rclpy.spin_once(self, timeout_sec=0.1)
        rclpy.spin_once(self, timeout_sec=0.05)
        return self.sim_ns

    def _rgb(self, m):
        self.rgb[self._key(m)] = m
        self.n_rgb += 1
        if len(self.rgb) > 10:
            self.rgb.pop(min(self.rgb))

    def _depth(self, m):
        self.depth[self._key(m)] = m
        if len(self.depth) > 10:
            self.depth.pop(min(self.depth))

    def fresh_pair(self, after_ns: int, timeout: float = 8.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
            keys = sorted(k for k in set(self.rgb) & set(self.depth) if k > after_ns)
            if keys:
                return self.rgb[keys[0]], self.depth[keys[0]]
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--world', default='showcase_forest')
    ap.add_argument('--world_sdf', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--scenes', type=int, default=300)
    ap.add_argument('--views', type=int, default=8)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--val_every', type=int, default=5, help='every Nth scene goes to val')
    ap.add_argument('--respawn_every', type=int, default=15, help='new random pool every N scenes')
    ap.add_argument('--pool', default='5,4,3', help='stumps,logs,bags in the pool')
    ap.add_argument('--negatives', type=float, default=0.1, help='share of scenes with no objects')
    ap.add_argument('--rate', type=float, default=10.0)
    ap.add_argument('--prefix', default='s')
    ap.add_argument('--ground', type=float, default=36.0, help='camera stays within +-this (m)')
    a = ap.parse_args()

    rng = random.Random(a.seed)
    trees = parse_tree_truth(a.world_sdf)
    xs = [t[1] for t in trees]
    ys = [t[2] for t in trees]
    oaks = [t for t in trees if 'oak' in t[0].lower()]
    bounds = (min(xs) - 3, max(xs) + 3, min(ys) - 3, max(ys) + 3)
    for split in ('train', 'val'):
        os.makedirs(f'{a.out}/images/{split}', exist_ok=True)
        os.makedirs(f'{a.out}/labels/{split}', exist_ok=True)
    with open(f'{a.out}/data.yaml', 'w') as fh:
        fh.write(f'path: {os.path.abspath(a.out)}\ntrain: images/train\nval: images/val\n'
                 f'names:\n' + ''.join(f'  {i}: {n}\n' for i, n in enumerate(CLASSES)))

    rclpy.init()
    node = Collector()
    svc = f'/world/{a.world}'
    tmpdir = tempfile.mkdtemp(prefix='yolo_collect_')

    def remove(name):
        ign_call(f'{svc}/remove', 'ignition.msgs.Entity', f'name: "{name}" type: MODEL')

    def spawn(name, sdf):
        remove(name)
        path = os.path.join(tmpdir, name + '.sdf')
        with open(path, 'w') as fh:
            fh.write(sdf)
        return ign_call(f'{svc}/create', 'ignition.msgs.EntityFactory',
                        f'sdf_filename: "{path}" name: "{name}"')

    if not spawn('datacam', datacam_sdf(a.rate)):
        sys.exit('could not spawn datacam')
    counts = [int(c) for c in a.pool.split(',')]
    pool = []
    import cv2

    def respawn_pool(gen):
        if pool:
            ign_call(f'{svc}/set_pose_vector', 'ignition.msgs.Pose_V',
                     ''.join(pose_req(ob['name'], 400, 400, -50) for ob in pool))
        for ob in pool:
            remove(ob['name'])
        pool.clear()
        for cls, n in enumerate(counts):
            for k in range(n):
                p = random_params(CLASSES[cls], rng)
                name = f'ev_{CLASSES[cls]}_{gen}_{k}'
                if spawn(name, evidence_sdf(name, 400 + 10 * len(pool), 400, 0.0, p)):
                    pool.append({'name': name, 'cls': cls, 'p': p})
        node.get_logger().info(f'pool {gen}: {len(pool)} objects')

    meta = open(f'{a.out}/meta.jsonl', 'a')
    n_img, n_box, t_start = 0, [0, 0, 0], time.time()
    for scene in range(a.scenes):
        if scene % a.respawn_every == 0:
            respawn_pool(scene // a.respawn_every + 1000 * a.seed)
        split = 'val' if scene % a.val_every == a.val_every - 1 else 'train'
        for _ in range(200):
            cx, cy = rng.uniform(bounds[0], bounds[1]), rng.uniform(bounds[2], bounds[3])
            if place_ok(cx, cy, 0.5, trees, []) and max(abs(cx), abs(cy)) < a.ground - 5:
                break
        negative = rng.random() < a.negatives
        chosen = [] if negative else rng.sample(pool, rng.randint(1, min(5, len(pool))))
        placed, objects, req = [], [], ''
        for ob in pool:
            ob['active'] = False
        for ob in chosen:
            rad = footprint_radius(ob['p'])
            for _ in range(60):
                ang, dist = rng.uniform(0, 2 * math.pi), rng.uniform(0, 4.5)
                x, y = cx + dist * math.cos(ang), cy + dist * math.sin(ang)
                if place_ok(x, y, rad, trees, placed, trunk_clear=0.3):
                    yaw = rng.uniform(-math.pi, math.pi)
                    placed.append((x, y, rad))
                    ob.update(active=True, x=x, y=y, yaw=yaw)
                    req += pose_req(ob['name'], x, y, model_z(ob['p']), 0, 0, yaw)
                    objects.append({'cls': ob['cls'], 'pts': surface_points(ob['p'], x, y, yaw),
                                    'name': ob['name']})
                    break
        for i, ob in enumerate(pool):
            if not ob['active']:
                req += pose_req(ob['name'], 400 + 10 * i, 400, -20)
        if not ign_call(f'{svc}/set_pose_vector', 'ignition.msgs.Pose_V', req):
            node.get_logger().warn(f'scene {scene}: objects not moved; skipping the scene')
            continue

        for view in range(a.views):
            for _ in range(100):
                ang, dist = rng.uniform(0, 2 * math.pi), rng.uniform(1.5, 14.0)
                x, y = cx + dist * math.cos(ang), cy + dist * math.sin(ang)
                if (place_ok(x, y, 0.5, trees, placed, trunk_clear=0.2)
                        and place_ok(x, y, 0.5, oaks, [], trunk_clear=1.8)
                        and max(abs(x), abs(y)) < a.ground):
                    break
            yaw = math.atan2(cy - y, cx - x) + rng.uniform(-0.8, 0.8)
            cam = (x, y, CAM_Z + rng.uniform(-0.04, 0.05), rng.uniform(-0.04, 0.04),
                   rng.uniform(-0.07, 0.07), yaw)
            if not ign_call(f'{svc}/set_pose', 'ignition.msgs.Pose', pose_fields('datacam', *cam)):
                node.get_logger().warn('camera not moved; skipping the view')
                continue
            pair = node.fresh_pair(node.sim_now() + 250_000_000)
            if pair is None:
                node.get_logger().warn('no camera frame; skipping view')
                continue
            rgb_msg, depth_msg = pair
            rgb = np.frombuffer(rgb_msg.data, np.uint8).reshape(H, W, 3)
            depth = np.frombuffer(depth_msg.data, np.float32).reshape(H, W)
            labels = label_objects(depth, cam, objects)
            stem = f'{a.prefix}{a.seed}_{scene:04d}_{view}'
            cv2.imwrite(f'{a.out}/images/{split}/{stem}.jpg', rgb[..., ::-1],
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(f'{a.out}/labels/{split}/{stem}.txt', 'w') as fh:
                for c, bx, by, bw, bh, vis, dmin in labels:
                    fh.write(f'{c} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}\n')
                    n_box[c] += 1
            meta.write(json.dumps({'stem': stem, 'split': split, 'cam': cam,
                                   'objects': [{'name': ob['name'], 'cls': ob['cls'],
                                                'x': ob['x'], 'y': ob['y'], 'yaw': ob['yaw'],
                                                'p': ob['p']} for ob in pool if ob['active']],
                                   'labels': labels}) + '\n')
            n_img += 1
        if scene % 10 == 9:
            meta.flush()
            el = time.time() - t_start
            node.get_logger().info(f'scene {scene + 1}/{a.scenes}: {n_img} images, boxes '
                                   f'stump {n_box[0]} log {n_box[1]} bag {n_box[2]}, '
                                   f'{el / max(n_img, 1):.2f} s/image; failed calls {FAILED or 0}')
    meta.close()
    node.get_logger().info(f'done: {n_img} images, boxes {n_box}')
    rclpy.shutdown()


if __name__ == '__main__':
    main()
