"""Project the drone camera onto the world grid from a bag (option A analysis).

usage: camera_species.py BAG WORLD [--bin-s 60] [--out FILE.npz]

Reads /parrot1/camera/image, /parrot1/camera/depth/image, /parrot1/odometry
and the last /forest_canopy_map. Canopy pixels are placed with
deforestation_monitoring.camera_species.project (the same code as the live
camera_species_mapper node) and accumulated per 0.25 m cell (the
scan_mapper grid) in time bins: pixel count, RGB sums and the per-pixel
blue/green sum. Also per bin: `high` = canopy pixels at >= --open-height
and `open` = rays that passed --open-height at the cell without a hit
(camera_species.open_rays), so high / (high + open) is the fraction of the
camera's view of a spot that is canopy. The npz feeds camera_eval.py,
camera_change.py and camera_pines.py.

Printed: camera height minus LiDAR canopy height (a check of the pose and
altitude) and the median B/G at oak and pine trunks.
"""
import argparse

import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import Image

from deforestation_monitoring.camera_species import open_rays, project, quat_to_matrix
from deforestation_monitoring.tree_detection import parse_tree_truth

RES, ORIGIN, DIM = 0.25, -40.0, 320          # scan_mapper grid


def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def read(bag, bin_s, altitude, open_height=3.0):
    r = SequentialReader()
    r.open(StorageOptions(uri=bag, storage_id='sqlite3'), ConverterOptions('cdr', 'cdr'))
    odoms, rgbs, depths, canopy = [], {}, {}, None
    while r.has_next():
        topic, data, _ = r.read_next()
        if topic == '/parrot1/odometry':
            m = deserialize_message(data, Odometry)
            odoms.append((stamp(m), m))
        elif topic == '/parrot1/camera/image':
            m = deserialize_message(data, Image)
            rgbs[round(stamp(m), 3)] = m
        elif topic == '/parrot1/camera/depth/image':
            m = deserialize_message(data, Image)
            depths[round(stamp(m), 3)] = m
        elif topic == '/forest_canopy_map':
            canopy = deserialize_message(data, OccupancyGrid)
    odom_t = np.array([t for t, _ in odoms])
    frames = sorted(set(rgbs) & set(depths))
    print(f'{len(frames)} RGB+depth frames, {len(odoms)} odometry')
    t0 = frames[0]
    nbins = int((frames[-1] - t0) // bin_s) + 1
    count = np.zeros((nbins, DIM, DIM), np.float32)
    rgb_sum = np.zeros((nbins, DIM, DIM, 3), np.float32)
    bg_sum = np.zeros((nbins, DIM, DIM), np.float32)
    high = np.zeros((nbins, DIM, DIM), np.float32)
    opened = np.zeros((nbins, DIM, DIM), np.float32)
    zsum = np.zeros((DIM, DIM), np.float32)
    zcount = np.zeros((DIM, DIM), np.float32)
    for t in frames:
        i = int(np.clip(np.searchsorted(odom_t, t), 1, len(odom_t) - 1))
        i = i if abs(odom_t[i] - t) < abs(odom_t[i - 1] - t) else i - 1
        if abs(odom_t[i] - t) > 0.1:
            continue
        m, d, o = rgbs[t], depths[t], odoms[i][1]
        rgb = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
        depth = np.frombuffer(d.data, np.float32).reshape(d.height, d.width)
        q = o.pose.pose.orientation
        rot = quat_to_matrix(q.x, q.y, q.z, q.w)
        p, px = project(rgb, depth, rot, o.pose.pose.position.x,
                        o.pose.pose.position.y, altitude)
        b = int((t - t0) // bin_s)
        xy = open_rays(depth, rot, o.pose.pose.position.x, o.pose.pose.position.y,
                       altitude, open_height)
        ox = ((xy[:, 0] - ORIGIN) / RES).astype(int)
        oy = ((xy[:, 1] - ORIGIN) / RES).astype(int)
        inb = (ox >= 0) & (ox < DIM) & (oy >= 0) & (oy < DIM)
        np.add.at(opened[b], (ox[inb], oy[inb]), 1.0)
        ix = ((p[:, 0] - ORIGIN) / RES).astype(int)
        iy = ((p[:, 1] - ORIGIN) / RES).astype(int)
        inb = (ix >= 0) & (ix < DIM) & (iy >= 0) & (iy < DIM)
        ix, iy, px, z = ix[inb], iy[inb], px[inb], p[inb, 2]
        np.add.at(count[b], (ix, iy), 1.0)
        tall = z >= open_height
        np.add.at(high[b], (ix[tall], iy[tall]), 1.0)
        np.add.at(bg_sum[b], (ix, iy), px[:, 2] / np.maximum(px[:, 1], 1.0))
        for ch in range(3):
            np.add.at(rgb_sum[b, :, :, ch], (ix, iy), px[:, ch])
        np.add.at(zsum, (ix, iy), z)
        np.add.at(zcount, (ix, iy), 1.0)
    chm = None
    if canopy is not None:
        g = np.array(canopy.data, np.int16).reshape(canopy.info.height, canopy.info.width).T
        chm = np.where(g >= 0, g / 10.0, np.nan).astype(np.float32)   # [ix, iy], metres
    return dict(count=count, rgb_sum=rgb_sum, bg_sum=bg_sum, high=high, open=opened,
                zsum=zsum, zcount=zcount,
                chm=chm if chm is not None else np.zeros(0), t0=t0, bin_s=bin_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag')
    ap.add_argument('world')
    ap.add_argument('--bin-s', type=float, default=60.0)
    ap.add_argument('--altitude', type=float, default=10.0)
    ap.add_argument('--open-height', type=float, default=3.0)
    ap.add_argument('--out')
    a = ap.parse_args()
    d = read(a.bag, a.bin_s, a.altitude, a.open_height)
    chm, zc = d['chm'], d['zcount']
    if chm.size:
        both = (zc > 0) & np.isfinite(chm) & (chm >= 1.5)
        dz = d['zsum'][both] / zc[both] - chm[both]
        print(f'camera height - LiDAR height on {both.sum()} canopy cells: median '
              f'{np.median(dz):+.2f} m, p10 {np.percentile(dz, 10):+.2f}, p90 {np.percentile(dz, 90):+.2f}')
    c, s = d['count'].sum(0), d['bg_sum'].sum(0)
    xs = ORIGIN + (np.arange(DIM) + 0.5) * RES
    gx, gy = np.meshgrid(xs, xs, indexing='ij')
    for kind in ('oak', 'pine'):
        v = []
        for name, x, y in parse_tree_truth(a.world):
            m = np.hypot(gx - x, gy - y) <= 0.75
            if name.startswith(kind) and c[m].sum() >= 20:
                v.append(s[m].sum() / c[m].sum())
        print(f'{kind}: {len(v)} trees seen, B/G at trunk p10 {np.percentile(v, 10):.2f} '
              f'median {np.median(v):.2f} p90 {np.percentile(v, 90):.2f}')
    if a.out:
        np.savez_compressed(a.out, **d)
        print('saved', a.out)


if __name__ == '__main__':
    main()
