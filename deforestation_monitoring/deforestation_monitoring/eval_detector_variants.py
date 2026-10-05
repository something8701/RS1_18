#!/usr/bin/env python3
"""A/B detector parameter variants on recorded live canopy maps.

For every variant (``name:key=value,key=value``, applied on top of
``config/tree_detection_params.yaml``) this runs ``detect_trees`` on frames
of the recorded surveys and prints, per world:

* sparse / showcase: TP/FP/FN at 1.0 m (one-to-one), the regression guard;
* dense: recall per removal tier (``visibility.removal_tier`` on each map)
  at 1.5 m and 2.0 m, closest-pair-first one-to-one, and the number of
  detections not paired to any trunk within 2.0 m (false crowns).

    ros2 run deforestation_monitoring eval_detector_variants \\
        base: lock:seed_lock=True snap:seed_lock=True,seed_snap=0.5
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple

import numpy as np
import yaml

from .tree_detection import (DetectionParams, detect_trees, match_detections,
                             parse_tree_truth)
from .visibility import classify_tree, removal_tier

HOME = os.path.expanduser('~/rs1_18_ws')
SRC = os.path.join(HOME, 'src/RS1_18')
# (bag, world, t_max s after the first map = just before any removal, box)
DATASETS = {
    'sparse': [('sparse_survey', 'sparse_trees', None, 15.0),
               ('removal_sparse1', 'sparse_trees', 125.0, 15.0)],
    'showcase': [('showcase_survey', 'showcase_forest', None, 30.0),
                 ('showcase_removal2', 'showcase_forest', 265.0, 30.0)],
    'dense': [('dense_survey', 'dense_forest', None, 30.0),
              ('removal4', 'dense_forest', 270.0, 30.0),
              ('removal6', 'dense_forest', 270.0, 30.0),
              ('removal8', 'dense_forest', 270.0, 30.0)],
}


def _cached_maps(bag: str, t_max, frames: int, cache_dir: str):
    path = os.path.join(cache_dir, f'{bag}_{t_max}_{frames}.npz')
    if os.path.exists(path):
        z = np.load(path)
        return list(z['maps']), float(z['res']), float(z['ox']), float(z['oy'])
    from .tune_on_recording import load_canopy_maps
    maps, res, ox, oy = load_canopy_maps(os.path.join(HOME, 'bags', bag),
                                         t_max, frames)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(path, maps=np.array(maps), res=res, ox=ox, oy=oy)
    return maps, res, ox, oy


def _pairs(dets, trees, radius):
    cand = sorted((math.hypot(d.x - t[1], d.y - t[2]), di, ti)
                  for di, d in enumerate(dets) for ti, t in enumerate(trees))
    used_d, used_t, out = set(), set(), {}
    for dist, di, ti in cand:
        if dist > radius:
            break
        if di in used_d or ti in used_t:
            continue
        used_d.add(di)
        used_t.add(ti)
        out[ti] = di
    return out


def evaluate(args) -> Tuple[str, Dict]:
    name, overrides, frames, cache_dir = args
    base = yaml.safe_load(open(os.path.join(
        SRC, 'deforestation_monitoring/config/tree_detection_params.yaml')))
    params = DetectionParams.from_dict(base['/**']['ros__parameters'])
    for k, v in overrides.items():
        cur = getattr(params, k)
        setattr(params, k, (v.lower() == 'true') if isinstance(cur, bool)
                else type(cur)(v))
    out: Dict = {}
    for group, sets in DATASETS.items():
        acc = {'tp': 0, 'fp': 0, 'fn': 0, 'false2': 0, 'dets': 0,
               'tiers': {}}
        for bag, world, t_max, half in sets:
            maps, res, ox, oy = _cached_maps(bag, t_max, frames, cache_dir)
            params.chm_resolution = res
            truth = parse_tree_truth(os.path.join(
                SRC, '41068_ignition_bringup/worlds', f'{world}.sdf'))
            inbox = [t for t in truth if abs(t[1]) <= half and abs(t[2]) <= half]
            for g in maps:
                scanned = g >= 0
                chm = np.where(scanned, g, 0.0).astype(np.float32)
                dets = [d for d in detect_trees(chm, scanned, params, ox, oy)
                        if abs(d.x) <= half and abs(d.y) <= half]
                acc['dets'] += len(dets)
                tp, fp, fn, _ = match_detections(dets, inbox, 1.0)
                acc['tp'] += tp
                acc['fp'] += fp
                acc['fn'] += fn
                if group != 'dense':
                    continue
                p15, p20 = _pairs(dets, inbox, 1.5), _pairs(dets, inbox, 2.0)
                acc['false2'] += len(dets) - len(p20)
                for ti, t in enumerate(inbox):
                    if abs(t[1]) > half - 2 or abs(t[2]) > half - 2:
                        continue
                    cls, _ = classify_tree(g, res, (ox, oy), t)
                    key = f"{removal_tier(t, cls, truth)}-{t[0].split('_')[0]}"
                    c = acc['tiers'].setdefault(key, [0, 0, 0])
                    c[0] += 1
                    c[1] += ti in p15
                    c[2] += ti in p20
        out[group] = acc
    return name, out


def _fmt(name: str, r: Dict) -> str:
    s, c, d = r['sparse'], r['showcase'], r['dense']
    f1 = 2 * c['tp'] / max(2 * c['tp'] + c['fp'] + c['fn'], 1)
    tiers = d['tiers']

    def rec(key):
        n, a, b = tiers.get(key, (0, 0, 0))
        return f'{a / max(n, 1):.2f}/{b / max(n, 1):.2f}'
    return (f"{name:18s} sparse {s['tp']}/{s['fp']}/{s['fn']} | showcase F1 "
            f"{f1:.3f} ({c['tp']}/{c['fp']}/{c['fn']}) | dense canopy oak "
            f"{rec('canopy-oak')} pine {rec('canopy-pine')} | crown-shared "
            f"{rec('crown-shared-pine')} | dense false@2m "
            f"{d['false2']}/{d['dets']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('variants', nargs='+', help='name:key=val,key=val')
    ap.add_argument('--frames', type=int, default=8)
    ap.add_argument('--cache', default=os.path.join(HOME, 'bags', '.map_cache'))
    ap.add_argument('--jobs', type=int, default=8)
    a = ap.parse_args(argv)
    jobs = []
    for spec in a.variants:
        name, _, rest = spec.partition(':')
        ov = dict(kv.split('=', 1) for kv in rest.split(',') if kv)
        jobs.append((name, ov, a.frames, a.cache))
    # Fill the cache once, serially (bag reading is the slow part).
    for sets in DATASETS.values():
        for bag, _, t_max, _ in sets:
            _cached_maps(bag, t_max, a.frames, a.cache)
    print('columns: sparse/showcase TP/FP/FN @1.0 m; dense recall @1.5/@2.0 m')
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for name, r in ex.map(evaluate, jobs):
            print(_fmt(name, r), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
