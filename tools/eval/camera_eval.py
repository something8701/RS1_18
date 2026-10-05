"""How well does camera colour separate pines from oaks? (option A feasibility)

usage: camera_eval.py MAP.npz WORLD

MAP.npz comes from camera_species.py (per-cell pixel counts and RGB sums).
Trees are tiered exactly as in removal_test (visibility.py on the LiDAR
canopy map in the npz, SDF trunks only). For each colour feature the AUC of
pine vs oak is printed:
  - per tree: mean colour within 0.75 m of the trunk;
  - per cell: cells within 0.5 m of a canopy-tier pine trunk vs cells within
    2.5 m of an oak trunk and more than 3 m from every pine.
AUC 0.5 = no information, 1.0 = perfect separation.
"""
import sys

import numpy as np

from deforestation_monitoring.tree_detection import parse_tree_truth
from deforestation_monitoring.visibility import classify_tree, removal_tier, species_of

RES, ORIGIN, DIM = 0.25, -40.0, 320
XS = ORIGIN + (np.arange(DIM) + 0.5) * RES


def features(rgb):
    """Colour features from mean RGB (..., 3)."""
    r, g, b = rgb[..., 0], rgb[..., 1], np.maximum(rgb[..., 2], 0)
    g = np.maximum(g, 1.0)
    mx, mn = rgb.max(-1), rgb.min(-1)
    return {
        'B/G': b / g,
        'R/G': r / g,
        'saturation': (mx - mn) / np.maximum(mx, 1.0),
        'brightness': g,
        'excess green': (2 * g - r - b) / np.maximum(r + g + b, 1.0),
    }


def auc(pos, neg):
    """P(pos > neg) by ranks; symmetric: returns max(auc, 1 - auc) and direction."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float('nan'), ''
    allv = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(allv)) + 1.0
    a = (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return (a, 'pine higher') if a >= 0.5 else (1 - a, 'pine lower')


def disc_mask(x, y, radius):
    gx, gy = np.meshgrid(XS, XS, indexing='ij')
    return np.hypot(gx - x, gy - y) <= radius


def main():
    d = np.load(sys.argv[1])
    truth = parse_tree_truth(sys.argv[2])
    count, rgb_sum, chm = d['count'].sum(0), d['rgb_sum'].sum(0), d['chm']
    grid = np.where(np.isfinite(chm), chm, -1.0)
    tiers = {t[0]: removal_tier(t, classify_tree(grid, RES, (ORIGIN, ORIGIN), t)[0], truth)
             for t in truth}
    in_box = {t[0] for t in truth if abs(t[1]) <= 30 and abs(t[2]) <= 30}

    # per tree
    per_tree = {}
    for name, x, y in truth:
        m = disc_mask(x, y, 0.75)
        n = count[m].sum()
        if n >= 20:
            per_tree[name] = rgb_sum[m].sum(0) / n
    groups = {}
    for name in per_tree:
        if name in in_box:
            key = 'oak' if species_of(name) == 'oak' else f'pine ({tiers[name]})'
            groups.setdefault(key, []).append(name)
    print('trees with camera coverage:', {k: len(v) for k, v in sorted(groups.items())})
    oaks = np.array([per_tree[n] for n in groups.get('oak', [])])
    print(f'{"feature":14s} ' + ' '.join(f'{k:>24s}' for k in sorted(groups)))
    fo = features(oaks)
    for f in fo:
        row = []
        for k in sorted(groups):
            v = features(np.array([per_tree[n] for n in groups[k]]))[f]
            row.append(f'{np.median(v):8.3f}')
        aucs = []
        for k in sorted(groups):
            if k.startswith('pine'):
                v = features(np.array([per_tree[n] for n in groups[k]]))[f]
                a, _ = auc(v, fo[f])
                aucs.append(f'{k.split("(")[1][:-1]} {a:.2f}')
        print(f'{f:14s} ' + ' '.join(f'{r:>24s}' for r in row) + '   AUC vs oak: ' + ', '.join(aucs))

    # per cell
    pine_cells = np.zeros((DIM, DIM), bool)
    near_pine = np.zeros((DIM, DIM), bool)
    oak_cells = np.zeros((DIM, DIM), bool)
    for name, x, y in truth:
        if species_of(name) == 'pine':
            near_pine |= disc_mask(x, y, 3.0)
            if tiers.get(name) == 'canopy':
                pine_cells |= disc_mask(x, y, 0.5)
        else:
            oak_cells |= disc_mask(x, y, 2.5)
    oak_cells &= ~near_pine
    seen = count >= 5
    mean = np.where(seen[..., None], rgb_sum / np.maximum(count, 1)[..., None], np.nan)
    fp, fo = features(mean[pine_cells & seen]), features(mean[oak_cells & seen])
    print(f'\nper cell: {int((pine_cells & seen).sum())} canopy-pine cells, '
          f'{int((oak_cells & seen).sum())} oak cells')
    for f in fp:
        a, direction = auc(fp[f], fo[f])
        print(f'  {f:14s} pine median {np.median(fp[f]):.3f}  oak median '
              f'{np.median(fo[f]):.3f}  AUC {a:.2f} ({direction})')


if __name__ == '__main__':
    main()
