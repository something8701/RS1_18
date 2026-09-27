#!/usr/bin/env python3
"""Tune the tree detector on recorded live canopy maps.

Calibration on synthetic maps (``calibrate_tree_detection``) did not carry
over to the live dense world, so this tool searches the detector parameters
on real ``/forest_canopy_map`` frames from rosbags:

* a guard recording (for example sparse_trees) must stay perfect: every
  frame must find all trees with 0 FP and 0 FN at ``--guard-radius``
  (1.0 m);
* on the target recording (for example dense_forest) F1 is maximised at
  ``--target-radius`` (1.5 m), optionally only for trees not near an oak
  (``--occlusion-filter``).

It uses random search, or local search around ``--seed`` (JSON), and writes
the best parameters merged into the params YAML (``--out``) plus a JSON
report.

    ros2 run deforestation_monitoring tune_on_recording \\
        --guard ~/rs1_18_ws/bags/sparse:sparse_trees \\
        --target ~/rs1_18_ws/bags/dense:dense_forest --target-tmax 285 \\
        --trials 300 --out /tmp/tuned.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from .removal_test import occluded_from_above
from .tree_detection import (DetectionParams, detect_trees, match_detections,
                             parse_tree_truth, params_to_ros_yaml)

SPACE: Dict[str, list] = {
    'smooth_sigma': [0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.1],
    'window_scale': [0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.6, 0.7],
    'window_offset': [0.2, 0.3, 0.5, 0.8],
    'max_window': [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    'min_sep': [1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
    'saddle_ratio': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    'overlap_threshold': [0.25, 0.5, 1.0, 1.5, 2.0, 3.0],
    'merge_small_area': [0.0, 1.0, 2.5],
    'merge_radius': [3.0, 5.0, 7.0],
    'min_crown_cells': [5, 10, 15],
    'merge_max_radius': [6.0, 7.0, 8.0],
    'position_mode': ['crown', 'band', 'auto'],
    'ground_margin': [0.3, 0.5, 0.8, 1.2],
    'min_height': [1.5, 2.0],
    'auto_threshold_cap': [1.5, 2.0, 2.5, 3.0],
}

WORLD_BOX = {
    'sparse_trees': (-15.0, 15.0, -15.0, 15.0),
    'cluster_test': (-10.0, 10.0, -10.0, 10.0),
}
DEFAULT_BOX = (-30.0, 30.0, -30.0, 30.0)


def load_canopy_maps(bag: str, t_max: Optional[float] = None,
                     frames: int = 15) -> Tuple[List[np.ndarray], float, float, float]:
    """Return ``frames`` evenly spaced maps from the last quarter of the bag
    (up to ``t_max`` s after the first map). Returns (maps [ix, iy] in metres,
    -1 = unscanned, resolution, origin_x, origin_y)."""
    from rclpy.serialization import deserialize_message
    from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
    from nav_msgs.msg import OccupancyGrid

    reader = SequentialReader()
    reader.open(StorageOptions(uri=bag, storage_id='sqlite3'),
                ConverterOptions('cdr', 'cdr'))
    raw, t0 = [], None
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        if topic != '/forest_canopy_map':
            continue
        t0 = stamp if t0 is None else t0
        if t_max is not None and (stamp - t0) / 1e9 > t_max:
            break
        raw.append(data)
    raw = raw[3 * len(raw) // 4:]
    idx = np.unique(np.linspace(0, len(raw) - 1, min(frames, len(raw))).astype(int))
    maps, info = [], None
    for i in idx:
        m = deserialize_message(raw[i], OccupancyGrid)
        info = m.info
        g = np.array(m.data, dtype=np.float32).reshape(
            m.info.height, m.info.width).T
        maps.append(np.where(g >= 0, g / 10.0, -1.0))
    return (maps, info.resolution, info.origin.position.x,
            info.origin.position.y)


def score(params: DetectionParams, maps, res, ox, oy, truth, box,
          radius: float) -> Dict:
    x0, x1, y0, y1 = box
    t_in = [t for t in truth if x0 <= t[1] <= x1 and y0 <= t[2] <= y1]
    p = DetectionParams.from_dict(params.to_dict())
    p.chm_resolution = res
    tp = fp = fn = 0
    counts = []
    for g in maps:
        scanned = g >= 0
        chm = np.where(scanned, g, 0.0).astype(np.float32)
        dets = [d for d in detect_trees(chm, scanned, p, ox, oy)
                if x0 <= d.x <= x1 and y0 <= d.y <= y1]
        counts.append(len(dets))
        a, b, c, _ = match_detections(dets, t_in, radius)
        tp, fp, fn = tp + a, fp + b, fn + c
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {'tp': tp, 'fp': fp, 'fn': fn,
            'f1': 2 * prec * rec / max(prec + rec, 1e-9),
            'precision': prec, 'recall': rec,
            'mean_count': float(np.mean(counts)) if counts else 0.0,
            'count_cv': float(np.std(counts) / max(np.mean(counts), 1e-9))
            if counts else 0.0}


def neighbour(cfg: Dict, rng: np.random.Generator, p_move: float = 0.35) -> Dict:
    out = dict(cfg)
    for k, opts in SPACE.items():
        if rng.random() >= p_move:
            continue
        cur = out.get(k)
        i = opts.index(cur) if cur in opts else int(rng.integers(len(opts)))
        out[k] = opts[int(np.clip(i + rng.choice([-1, 1]), 0, len(opts) - 1))]
    return out


def _split(spec: str) -> Tuple[str, str]:
    path, world = spec.rsplit(':', 1)
    return os.path.expanduser(path), world


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--guard', required=True, help='BAG:WORLD that must stay perfect')
    ap.add_argument('--target', required=True, help='BAG:WORLD to maximise')
    ap.add_argument('--target-tmax', type=float, default=None,
                    help='use target maps up to this many s (e.g. before a removal)')
    ap.add_argument('--guard-tmax', type=float, default=None,
                    help='use guard maps up to this many s')
    ap.add_argument('--occlusion-filter', action='store_true',
                    help='score the target only against trees not near an '
                         'oak (removal_test.occluded_from_above)')
    ap.add_argument('--worlds-dir', default=None)
    ap.add_argument('--params', default=None, help='base params YAML')
    ap.add_argument('--trials', type=int, default=200)
    ap.add_argument('--seed', default=None, help='JSON config for local search')
    ap.add_argument('--rng', type=int, default=11)
    ap.add_argument('--guard-radius', type=float, default=1.0)
    ap.add_argument('--target-radius', type=float, default=1.5)
    ap.add_argument('--out', default='/tmp/deforestation_eval/tuned_params.yaml')
    ap.add_argument('--report', default='/tmp/deforestation_eval/tuning_report.json')
    args = ap.parse_args(argv)

    from ament_index_python.packages import get_package_share_directory
    worlds = args.worlds_dir or os.path.join(
        get_package_share_directory('41068_ignition_bringup'), 'worlds')
    params_path = args.params or os.path.join(
        get_package_share_directory('deforestation_monitoring'),
        'config', 'tree_detection_params.yaml')
    base = yaml.safe_load(open(params_path))['/**']['ros__parameters']

    g_bag, g_world = _split(args.guard)
    t_bag, t_world = _split(args.target)
    g_maps, g_res, g_ox, g_oy = load_canopy_maps(g_bag, args.guard_tmax)
    t_maps, t_res, t_ox, t_oy = load_canopy_maps(t_bag, args.target_tmax)
    g_truth = parse_tree_truth(os.path.join(worlds, f'{g_world}.sdf'))
    t_all = parse_tree_truth(os.path.join(worlds, f'{t_world}.sdf'))
    hidden = occluded_from_above(t_all) if args.occlusion_filter else set()
    t_truth = [t for t in t_all if t[0] not in hidden]
    g_box = WORLD_BOX.get(g_world, DEFAULT_BOX)
    t_box = WORLD_BOX.get(t_world, DEFAULT_BOX)
    print(f'guard {g_world}: {len(g_maps)} maps; target {t_world}: '
          f'{len(t_maps)} maps, {len(t_truth)} visible of {len(t_all)} trees',
          flush=True)

    rng = np.random.default_rng(args.rng)
    seed = json.loads(args.seed) if args.seed else None
    configs = [{}] + ([seed] if seed else [])
    while len(configs) < args.trials + 1:
        configs.append(neighbour(seed, rng) if seed else
                       {k: v[int(rng.integers(len(v)))] for k, v in SPACE.items()})

    results = []
    t_start = time.time()
    for i, cfg in enumerate(configs):
        params = DetectionParams.from_dict({**base, **cfg})
        params.min_crown_cells = int(params.min_crown_cells)
        g = score(params, g_maps, g_res, g_ox, g_oy, g_truth, g_box,
                  args.guard_radius)
        guard_ok = g['fp'] == 0 and g['fn'] == 0
        t = score(params, t_maps, t_res, t_ox, t_oy, t_truth, t_box,
                  args.target_radius) if guard_ok or i == 0 else None
        results.append({'config': cfg, 'guard': g, 'target': t,
                        'guard_ok': guard_ok})
        if t is not None:
            print(f'[{i}] guard {g["tp"]}/{g["fp"]}/{g["fn"]} | target F1 '
                  f'{t["f1"]:.3f} P {t["precision"]:.2f} R {t["recall"]:.2f} '
                  f'n {t["mean_count"]:.0f} cv {t["count_cv"]:.2f} | {cfg}',
                  flush=True)

    ok = sorted((r for r in results if r['guard_ok'] and r['target']),
                key=lambda r: -r['target']['f1'])
    print(f'\n{len(ok)}/{len(results)} configs keep {g_world} perfect '
          f'({time.time() - t_start:.0f} s)', flush=True)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, 'w') as fh:
        json.dump({'guard': args.guard, 'target': args.target,
                   'results': results}, fh, indent=1, default=str)
    if not ok:
        print('no configuration kept the guard world perfect')
        return 1
    best = ok[0]
    merged = DetectionParams.from_dict({**base, **best['config']})
    with open(args.out, 'w') as fh:
        fh.write(params_to_ros_yaml(merged))
    print('BEST', json.dumps({'target': best['target'], 'config': best['config']}))
    print(f'wrote {args.out} and {args.report}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
