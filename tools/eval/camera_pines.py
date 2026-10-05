"""Can camera colour find pines the height map misses? (option A)

usage: camera_pines.py MAP.npz WORLD [--mode spires|patches] [--min-bg 0.6]

MAP.npz comes from camera_species.py.
  spires (default): local maxima of the LiDAR canopy map (window
      --window m, >= --min-top m) whose camera colour within 0.5 m has mean
      per-pixel B/G >= min_bg. A pine is a cone with its tip over the trunk;
      the height map finds the tip but cannot tell it from an oak branch
      clump, the colour can.
  patches: 8-connected pine-coloured cells (B/G >= min_bg, camera height
      >= min_height), >= min_cells cells, at their pixel-weighted centroid.
Candidates are matched one-to-one to SDF trunks within 1.0 m and scored per
tier (visibility.py on the npz's LiDAR map), so the question "does colour
find the crown-shared pines?" gets a number, as well as how many
candidates sit on no pine at all.
"""
import argparse
import math

import numpy as np
from scipy import ndimage

from deforestation_monitoring.tree_detection import parse_tree_truth
from deforestation_monitoring.visibility import classify_tree, removal_tier, species_of

RES, ORIGIN, DIM = 0.25, -40.0, 320
XS = ORIGIN + (np.arange(DIM) + 0.5) * RES


def candidates(d, min_bg, min_cells, min_height):
    count, bg_sum = d['count'].sum(0), d['bg_sum'].sum(0)
    z = np.where(d['zcount'] > 0, d['zsum'] / np.maximum(d['zcount'], 1), 0)
    bg = np.where(count >= 5, bg_sum / np.maximum(count, 1), 0)
    mask = (bg >= min_bg) & (z >= min_height)
    labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
    out = []
    for i in range(1, n + 1):
        ix, iy = np.nonzero(labels == i)
        if len(ix) < min_cells:
            continue
        w = count[ix, iy]
        out.append((float((XS[ix] * w).sum() / w.sum()), float((XS[iy] * w).sum() / w.sum()),
                    len(ix)))
    return out


def spires(d, min_bg, window, min_top):
    chm = np.where(np.isfinite(d['chm']), d['chm'], 0.0)
    count, bg_sum = d['count'].sum(0), d['bg_sum'].sum(0)
    k = max(3, int(round(window / RES)) | 1)
    peak = (chm == ndimage.maximum_filter(chm, size=k)) & (chm >= min_top)
    gx, gy = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3), indexing='ij')
    disc = gx ** 2 + gy ** 2 <= 4                     # 0.5 m radius
    out = []
    for ix, iy in zip(*np.nonzero(peak)):
        if not (2 <= ix < DIM - 2 and 2 <= iy < DIM - 2):
            continue
        n = count[ix - 2:ix + 3, iy - 2:iy + 3][disc].sum()
        if n >= 20 and bg_sum[ix - 2:ix + 3, iy - 2:iy + 3][disc].sum() / n >= min_bg:
            out.append((float(XS[ix]), float(XS[iy]), 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz')
    ap.add_argument('world')
    ap.add_argument('--mode', choices=('spires', 'patches'), default='spires')
    ap.add_argument('--min-bg', type=float, default=0.6)
    ap.add_argument('--window', type=float, default=1.25)
    ap.add_argument('--min-top', type=float, default=3.0)
    ap.add_argument('--min-cells', type=int, default=6)
    ap.add_argument('--min-height', type=float, default=2.5)
    a = ap.parse_args()
    d = np.load(a.npz)
    truth = [t for t in parse_tree_truth(a.world) if abs(t[1]) <= 30 and abs(t[2]) <= 30]
    grid = np.where(np.isfinite(d['chm']), d['chm'], -1.0)
    tiers = {t[0]: removal_tier(t, classify_tree(grid, RES, (ORIGIN, ORIGIN), t)[0], truth)
             for t in truth}
    found = (spires(d, a.min_bg, a.window, a.min_top) if a.mode == 'spires'
             else candidates(d, a.min_bg, a.min_cells, a.min_height))
    cands = [c for c in found if abs(c[0]) <= 30 and abs(c[1]) <= 30]
    pairs = sorted((math.hypot(c[0] - t[1], c[1] - t[2]), ci, ti)
                   for ci, c in enumerate(cands) for ti, t in enumerate(truth)
                   if math.hypot(c[0] - t[1], c[1] - t[2]) <= 1.0)
    used_c, used_t, matched = set(), set(), {}
    for dist, ci, ti in pairs:
        if ci not in used_c and ti not in used_t:
            used_c.add(ci)
            used_t.add(ti)
            matched[truth[ti][0]] = dist
    print(f'{len(cands)} pine candidates ({a.mode}, B/G >= {a.min_bg})')
    for tier in ('canopy', 'crown-shared', 'understory'):
        names = [t[0] for t in truth if species_of(t[0]) == 'pine' and tiers[t[0]] == tier]
        hit = [n for n in names if n in matched]
        err = np.median([matched[n] for n in hit]) if hit else float('nan')
        print(f'  pines, {tier:12s}: found {len(hit):3d} of {len(names):3d}  '
              f'(median error {err:.2f} m)')
    oak_hits = [t[0] for t in truth if t[0] in matched and species_of(t[0]) == 'oak']
    unmatched = len(cands) - len(used_c)
    near_pine = sum(1 for ci, c in enumerate(cands) if ci not in used_c and any(
        math.hypot(c[0] - t[1], c[1] - t[2]) <= 2.5 for t in truth if species_of(t[0]) == 'pine'))
    print(f'  matched to an oak trunk: {len(oak_hits)}; on no trunk within 1 m: {unmatched} '
          f'({near_pine} of them within 2.5 m of a pine)')


if __name__ == '__main__':
    main()
