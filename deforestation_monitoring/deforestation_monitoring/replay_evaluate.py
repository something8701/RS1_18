#!/usr/bin/env python3
"""Offline replay evaluation: rebuild the CHM from the raw /parrot1/scan.

Reads a rosbag with ``/parrot1/scan`` and ``/parrot1/odometry`` and rebuilds
the canopy height model with the same projection as the live scan_mapper.
It does not use the recorded canopy map or terrain cloud, so the detector
can be tuned on the drone's real sensor geometry.

Metrics against the world SDF: precision, recall, F1, treetop movement
between the two halves of the bag, and false change events between the two
halves. For a run with no changes the last one must be 0.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import yaml

from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid

from .scan_mapper import project_scan_to_world

from .tree_detection import (
    DetectionParams, detect_trees, parse_tree_truth, score_detections)


def scan_to_world(scan: LaserScan, odom: Odometry, altitude: float):
    """The projection from scan_mapper (imported, not copied)."""
    ranges = np.array(scan.ranges, dtype=np.float64)
    angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
    valid = (np.isfinite(ranges) & (ranges >= 0.5) & (ranges <= scan.range_max))
    r, a = ranges[valid], angles[valid]
    if len(r) == 0:
        return np.zeros((0, 3))
    pos = np.array([
        odom.pose.pose.position.x,
        odom.pose.pose.position.y,
        0.0,
    ])
    return project_scan_to_world(
        r, a, odom.pose.pose.orientation, pos, altitude,
        pitch=math.pi / 2.0, yaw=0.0)


def scan_times(reader):
    """Bag receive times (ns) of all /parrot1/scan messages."""
    times = []
    while reader.has_next():
        topic, _, t = reader.read_next()
        if topic == '/parrot1/scan':
            times.append(t)
    return times


def accumulate(reader, altitude, res=0.25, origin=-40.0, dim=320,
               split_ns=None):
    """Rebuild the CHM. Scans before ``split_ns`` go to half 0 (pass A),
    the rest to half 1 (pass B). With ``split_ns=None`` scans alternate,
    which is not a pass A / pass B comparison."""
    height = np.zeros((dim, dim), dtype=np.float32)
    # Latest reading per cell, like scan_mapper.height_recent (the live
    # /forest_canopy_map): each scan overwrites the cells it hits with that
    # scan's max. The CHM itself uses the all-time max.
    recent = np.zeros((dim, dim), dtype=np.float32)
    hits = np.zeros((dim, dim), dtype=np.int32)
    last_odom = None
    halves = [np.zeros((dim, dim), dtype=np.float32),
              np.zeros((dim, dim), dtype=np.float32)]
    half_hits = [np.zeros((dim, dim), dtype=np.int32),
                 np.zeros((dim, dim), dtype=np.int32)]
    last_map = None
    order = 0
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic == '/parrot1/odometry':
            last_odom = deserialize_message(data, Odometry)
        elif topic == '/forest_canopy_map':
            last_map = deserialize_message(data, OccupancyGrid)
        elif topic == '/parrot1/scan' and last_odom is not None:
            scan = deserialize_message(data, LaserScan)
            pts = scan_to_world(scan, last_odom, altitude)
            if len(pts) == 0:
                continue
            ix = ((pts[:, 0] - origin) / res).astype(np.int64)
            iy = ((pts[:, 1] - origin) / res).astype(np.int64)
            ok = (ix >= 0) & (ix < dim) & (iy >= 0) & (iy < dim)
            ix, iy, zs = ix[ok], iy[ok], pts[ok, 2]
            if len(ix) == 0:
                continue
            half = (order % 2) if split_ns is None else int(t_ns >= split_ns)
            np.maximum.at(height, (ix, iy), zs)
            scan_max = np.full((dim, dim), -np.inf, dtype=np.float32)
            np.maximum.at(scan_max, (ix, iy), zs)
            touched = np.isfinite(scan_max)       # same rule as scan_mapper
            recent[touched] = np.maximum(scan_max[touched], 0.0)
            np.add.at(hits, (ix, iy), 1)
            np.maximum.at(halves[half], (ix, iy), zs)
            np.add.at(half_hits[half], (ix, iy), 1)
            order += 1
    return height, hits, halves, half_hits, last_map, recent


def match_passes(a, b, fallback_radius=0.6):
    """Match pass A treetops to pass B treetops one-to-one.

    Returns ``(unmatched_a, unmatched_b, matched_distances)``.
    """
    from scipy.optimize import linear_sum_assignment
    if not a or not b:
        return len(a), len(b), []
    pa, pb = np.asarray(a, float), np.asarray(b, float)
    if len(pa) > 1:
        d_aa = np.hypot(pa[:, None, 0] - pa[None, :, 0],
                        pa[:, None, 1] - pa[None, :, 1])
        np.fill_diagonal(d_aa, np.inf)
        radius = np.minimum(0.4 * d_aa.min(axis=1), 1.5)
    else:
        radius = np.array([fallback_radius])
    cost = np.hypot(pa[:, None, 0] - pb[None, :, 0],
                    pa[:, None, 1] - pb[None, :, 1])
    big = 1e6
    matrix = np.where(cost <= radius[:, None], cost, big)
    rows, cols = linear_sum_assignment(matrix)
    dists = [float(matrix[r, c]) for r, c in zip(rows, cols)
             if matrix[r, c] < big]
    return len(pa) - len(dists), len(pb) - len(dists), dists


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', required=True)
    parser.add_argument('--world', required=True)
    parser.add_argument('--params', required=True)
    parser.add_argument('--altitude', type=float, default=10.0)
    parser.add_argument('--match-radius', type=float, default=1.0)
    parser.add_argument(
        '--split', choices=('time', 'interleave'), default='time',
        help="'time': pass A = first half of the bag's scans in time, "
             "pass B = second half (a real two-pass comparison). "
             "'interleave': even vs odd scans (old behaviour).")
    parser.add_argument(
        '--split-sec', type=float, default=None,
        help='Seconds from the first scan at which pass B starts '
             '(default: the median scan time).')
    args = parser.parse_args(argv)

    def open_reader():
        r = SequentialReader()
        r.open(
            StorageOptions(uri=args.bag, storage_id='sqlite3'),
            ConverterOptions(
                input_serialization_format='cdr',
                output_serialization_format='cdr'))
        return r

    split_ns = None
    if args.split == 'time':
        times = scan_times(open_reader())
        if not times:
            print('no /parrot1/scan messages in bag')
            return 1
        if args.split_sec is not None:
            split_ns = times[0] + int(args.split_sec * 1e9)
        else:
            split_ns = int(np.median(times))
        print(f'pass split at +{(split_ns - times[0]) / 1e9:.1f}s '
              f'of {(times[-1] - times[0]) / 1e9:.1f}s '
              f'({len(times)} scans)')
    height, hits, halves, half_hits, last_map, recent = accumulate(
        open_reader(), args.altitude, res=0.25, split_ns=split_ns)
    print(f'scanned cells: {np.count_nonzero(hits)}')

    # Compare the rebuilt CHM with the live recorded canopy map on cells
    # both have observed. They should be almost the same, since both use
    # the same projection code.
    if last_map is not None:
        # OccupancyGrid data is row-major [iy, ix]; these grids are [ix, iy].
        live = np.array(last_map.data, dtype=np.float32).reshape(
            last_map.info.height, last_map.info.width).T
        common = (hits > 0) & (live >= 0)
        if np.any(common):
            # The live map is the latest reading per cell, not the max.
            diff = np.abs(live[common] / 10.0 - recent[common])
            print(f'parity vs live map: max={diff.max():.3f}m '
                  f'mean={diff.mean():.3f}m over {common.sum()} cells')

    params = DetectionParams.from_dict(
        yaml.safe_load(open(args.params))['/**']['ros__parameters'])
    params.chm_resolution = 0.25
    truth = parse_tree_truth(args.world)

    scanned = hits > 0
    dets = detect_trees(height, scanned, params, -40.0, -40.0)
    m = score_detections(dets, truth, args.match_radius)
    print('FULL BAG:', {k: round(v, 3) for k, v in m.items()})

    # Position error of the top-band centroid vs the single highest cell,
    # measured on true positives in tight groups only.
    from .tree_detection import match_detections
    for band in (0.5, 0.0):
        p = DetectionParams.from_dict(params.to_dict())
        p.centroid_band = band
        d2 = detect_trees(height, scanned, p, -40.0, -40.0)
        _, _, _, pairs = match_detections(d2, truth, args.match_radius)
        if pairs:
            errs = [d for _, _, d in pairs]
            errs.sort()
            print(f'centroid band={band}m: matched={len(pairs)} '
                  f'median err={np.median(errs):.3f}m p95={np.percentile(errs, 95):.3f}m')

    half_dets = []
    for half in range(2):
        h_scanned = half_hits[half] > 0
        d = detect_trees(halves[half], h_scanned, params, -40.0, -40.0)
        half_dets.append(d)
        hm = score_detections(d, truth, args.match_radius)
        print(f'HALF {half}:', {k: round(v, 3) for k, v in hm.items()})

    # The pass A vs pass B change check only uses cells seen in both passes.
    common = (half_hits[0] > 0) & (half_hits[1] > 0)

    def observed(dets):
        out = []
        for d in dets:
            ix = int((d.x + 40.0) / 0.25)
            iy = int((d.y + 40.0) / 0.25)
            if 0 <= ix < 320 and 0 <= iy < 320 and common[ix, iy]:
                out.append((d.x, d.y))
        return out

    # Same matching as the live tracker: one-to-one, radius
    # min(0.4 * nearest neighbour distance, 1.5 m), so a few cm of treetop
    # jitter is not counted as one LOST plus one GAINED.
    only_a, only_b, shifts = match_passes(
        observed(half_dets[0]), observed(half_dets[1]))
    flicker = only_a + only_b
    if shifts:
        print(f'matched treetop shift A->B: median={np.median(shifts):.3f}m '
              f'max={max(shifts):.3f}m (n={len(shifts)})')
    print(f'flicker between halves: {flicker} '
          f'(lost={only_a}, gained={only_b})')
    print('NO-CHANGE GATE:', 'PASS' if flicker == 0 else 'FAIL')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
