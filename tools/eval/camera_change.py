"""Does a cut pine show up as a colour change? (option A, change cue)

usage: camera_change.py MAP.npz WORLD [--bag BAG --log LAUNCH_LOG]

MAP.npz comes from camera_species.py (time bins of per-cell B/G sums). With
--bag/--log (a removal test), the removal moment and the removed trees are
read from the launch log ("[test] removed N: ..."); the wall-clock time is
converted to sim time with the bag's odometry (receive time vs header).

Per tree, B/G within 0.75 m of the trunk is computed per time bin (bins with
>= MIN_PIX pixels). Printed:
  - removed trees: B/G before vs after the cut;
  - standing pines (B/G before >= PINE): how often a single bin, or two
    consecutive bins, drop below OAK. That is the false-alarm risk of a
    "pine colour gone" rule.
  - canopy fraction (npz with `high`/`open`): per tree, the share of camera
    rays at the trunk (0.75 m) that hit canopy at >= 3 m, before vs after
    the cut, and how often a standing tree's fraction drops below FRAC in
    one bin or two consecutive bins (the false-alarm risk of a "the camera
    now sees through it" rule).
"""
import argparse
import gzip
import re

import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from nav_msgs.msg import Odometry

from deforestation_monitoring.tree_detection import parse_tree_truth

RES, ORIGIN, DIM = 0.25, -40.0, 320
XS = ORIGIN + (np.arange(DIM) + 0.5) * RES
MIN_PIX, PINE, OAK, FRAC = 20, 0.55, 0.50, 0.3


def removal(log_path, bag):
    opener = gzip.open if log_path.endswith('.gz') else open
    txt = opener(log_path, 'rt', errors='replace').read()
    m = re.search(r'\[(\d+\.\d+)\] \[removal_test\]: \[test\] removed \d+: (.*)', txt)
    wall, names = float(m.group(1)), [n.strip() for n in m.group(2).split(',')]
    r = SequentialReader()
    r.open(StorageOptions(uri=bag, storage_id='sqlite3'), ConverterOptions('cdr', 'cdr'))
    best = None
    while r.has_next():
        topic, data, ts = r.read_next()
        if topic == '/parrot1/odometry' and abs(ts * 1e-9 - wall) < 1.0:
            o = deserialize_message(data, Odometry)
            sim = o.header.stamp.sec + o.header.stamp.nanosec * 1e-9
            best = sim + (wall - ts * 1e-9)
            break
    return best, names


def series(count, bg_sum, x, y, radius=0.75):
    """Per-bin mean of per-pixel B/G within `radius` of (x, y), and pixel counts."""
    gx, gy = np.meshgrid(XS, XS, indexing='ij')
    m = np.hypot(gx - x, gy - y) <= radius
    n = count[:, m].sum(1)
    bg = np.where(n >= MIN_PIX, bg_sum[:, m].sum(1) / np.maximum(n, 1), np.nan)
    return bg, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz')
    ap.add_argument('world')
    ap.add_argument('--bag')
    ap.add_argument('--log')
    a = ap.parse_args()
    d = np.load(a.npz)
    count, bg_sum, t0, bin_s = d['count'], d['bg_sum'], float(d['t0']), float(d['bin_s'])
    truth = [t for t in parse_tree_truth(a.world) if abs(t[1]) <= 30 and abs(t[2]) <= 30]
    t_cut, removed = (None, [])
    if a.bag and a.log:
        t_cut, removed = removal(a.log, a.bag)
        print(f'cut at sim {t_cut:.0f} s (bin {(t_cut - t0) / bin_s:.1f}); removed: {", ".join(removed)}')
    cut_bin = int((t_cut - t0) // bin_s) if t_cut else count.shape[0]

    def pooled(bg, n, sl):
        w = np.where(np.isfinite(bg[sl]), n[sl], 0)
        return float(np.nansum(bg[sl] * w) / w.sum()) if w.sum() else float('nan'), int(w.sum())

    if removed:
        print(f'\n{"removed tree":14s} {"B/G before":>11s} {"B/G after":>10s}  pixels before/after')
        for name, x, y in truth:
            if name in removed:
                bg, n = series(count, bg_sum, x, y)
                b, nb = pooled(bg, n, slice(0, cut_bin))
                af, na = pooled(bg, n, slice(cut_bin + 1, None))
                flag = ''
                if name.startswith('pine'):
                    flag = ('colour change' if b >= PINE and af < OAK else
                            'no colour change' if b >= PINE else 'not pine-coloured before')
                print(f'{name:14s} {b:11.2f} {af:10.2f}  {nb:6d}/{na:<6d} {flag}')

    single = double = bins = pines = 0
    worst = []
    for name, x, y in truth:
        if not name.startswith('pine') or name in removed:
            continue
        bg, n = series(count, bg_sum, x, y)
        b, _ = pooled(bg, n, slice(0, max(cut_bin, 1)))
        if not b >= PINE:
            continue
        pines += 1
        v = bg[np.isfinite(bg)]
        bins += len(v)
        low = v < OAK
        single += int(low.sum())
        double += int(np.any(low[1:] & low[:-1]))
        if low.any():
            worst.append((name, round(float(v.min()), 2), int(low.sum()), len(v)))
    print(f'\nstanding pine-coloured pines: {pines}; bins below {OAK}: {single} of {bins}; '
          f'pines with 2 consecutive low bins: {double}')
    for w in sorted(worst, key=lambda w: w[1])[:10]:
        print('  ', w)
    if 'open' in d.files:
        canopy_fraction(d, truth, removed, cut_bin)


def fraction_series(high, opened, x, y, radius=0.75):
    gx, gy = np.meshgrid(XS, XS, indexing='ij')
    m = np.hypot(gx - x, gy - y) <= radius
    h, o = high[:, m].sum(1), opened[:, m].sum(1)
    return np.where(h + o >= MIN_PIX, h / np.maximum(h + o, 1), np.nan), h + o


def canopy_fraction(d, truth, removed, cut_bin):
    high, opened = d['high'], d['open']

    def pooled(f, n, sl):
        w = np.where(np.isfinite(f[sl]), n[sl], 0)
        return float(np.nansum(f[sl] * w) / w.sum()) if w.sum() else float('nan')

    if removed:
        print(f'\n{"removed tree":14s} canopy fraction before -> after (first bin after, all after)')
        for name, x, y in truth:
            if name in removed:
                f, n = fraction_series(high, opened, x, y)
                after = f[cut_bin + 1:]
                first = next((round(float(v), 2) for v in after if np.isfinite(v)), None)
                print(f'{name:14s} {pooled(f, n, slice(0, cut_bin)):.2f} -> '
                      f'{first}, {pooled(f, n, slice(cut_bin + 1, None)):.2f}')
    for kind in ('pine', 'oak'):
        trees = low1 = low2 = bins = 0
        worst = []
        for name, x, y in truth:
            if not name.startswith(kind) or name in removed:
                continue
            f, n = fraction_series(high, opened, x, y)
            before = pooled(f, n, slice(0, max(cut_bin, 1)))
            if not before >= 0.6:                  # a visible canopy tree
                continue
            v = f[np.isfinite(f)]
            trees += 1
            bins += len(v)
            low = v < FRAC
            low1 += int(low.sum())
            low2 += int(np.any(low[1:] & low[:-1]))
            if low.any():
                worst.append((name, round(float(v.min()), 2), int(low.sum()), len(v)))
        print(f'standing {kind}s with canopy fraction >= 0.6: {trees}; bins below {FRAC}: '
              f'{low1} of {bins}; trees with 2 consecutive low bins: {low2}')
        for w in sorted(worst, key=lambda w: w[1])[:6]:
            print('  ', w)


if __name__ == '__main__':
    main()
