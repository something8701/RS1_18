#!/usr/bin/env python3
# visibility.py: Detector-independent "visible from above" classification
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

Tree = Tuple[str, float, float]

DEFAULT_SPECIES_HEIGHT = {'oak': 6.2, 'pine': 4.6}


def species_of(name: str) -> str:
    return name.split('_')[0]


def disc_max(grid: np.ndarray, res: float, origin: Tuple[float, float],
             x: float, y: float, radius: float) -> Optional[float]:
    ix = int((x - origin[0]) / res)
    iy = int((y - origin[1]) / res)
    k = int(math.ceil(radius / res))
    x0, x1 = max(0, ix - k), min(grid.shape[0], ix + k + 1)
    y0, y1 = max(0, iy - k), min(grid.shape[1], iy + k + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    sub = grid[x0:x1, y0:y1]
    xx, yy = np.mgrid[x0:x1, y0:y1]
    inside = (xx - ix) ** 2 + (yy - iy) ** 2 <= (radius / res) ** 2
    vals = sub[inside & (sub >= 0)]
    return float(vals.max()) if vals.size else None


def species_top_heights(grid, res, origin, trees: Iterable[Tree],
                        radius: float = 0.75) -> Dict[str, float]:
    tops: Dict[str, list] = {}
    for name, x, y in trees:
        top = disc_max(grid, res, origin, x, y, radius)
        if top is not None:
            tops.setdefault(species_of(name), []).append(top)
    return {k: float(np.median(v)) for k, v in tops.items()}


def classify_tree(grid, res, origin, tree: Tree,
                  heights: Dict[str, float] = DEFAULT_SPECIES_HEIGHT,
                  top_radius: float = 0.75, above: float = 0.4,
                  below: float = 1.0) -> Tuple[str, Optional[float]]:
    name, x, y = tree
    top = disc_max(grid, res, origin, x, y, top_radius)
    h = heights.get(species_of(name))
    if top is None or h is None or top < h - below:
        return 'unobserved', top
    if top > h + above:
        return 'understory', top
    return 'visible', top


def classify_trees(grid, res, origin, trees: Sequence[Tree],
                   heights: Dict[str, float] = DEFAULT_SPECIES_HEIGHT,
                   **kw) -> Dict[str, Tuple[str, Optional[float]]]:
    return {t[0]: classify_tree(grid, res, origin, t, heights, **kw)
            for t in trees}


CROWN_SHARE_RADIUS = 5.5


def removal_tier(tree: Tree, visibility_class: str, truth: Sequence[Tree],
                 share_radius: float = CROWN_SHARE_RADIUS) -> str:
    if visibility_class != 'visible':
        return visibility_class
    name, x, y = tree
    if species_of(name) == 'pine' and any(
            species_of(o[0]) == 'oak'
            and math.hypot(o[1] - x, o[2] - y) <= share_radius
            for o in truth):
        return 'crown-shared'
    return 'canopy'


def top_drop(before: np.ndarray, now: np.ndarray, res: float,
             origin: Tuple[float, float], x: float, y: float,
             radius: float = 1.0, mode: str = 'cells',
             band: float = 0.3) -> float:
    ix = int((x - origin[0]) / res)
    iy = int((y - origin[1]) / res)
    k = int(math.ceil(radius / res))
    x0, x1 = max(0, ix - k), min(before.shape[0], ix + k + 1)
    y0, y1 = max(0, iy - k), min(before.shape[1], iy + k + 1)
    if x0 >= x1 or y0 >= y1:
        return 0.0
    b = before[x0:x1, y0:y1]
    n = now[x0:x1, y0:y1]
    xx, yy = np.mgrid[x0:x1, y0:y1]
    inside = (xx - ix) ** 2 + (yy - iy) ** 2 <= (radius / res) ** 2
    seen_b = inside & (b >= 0)
    if not seen_b.any():
        return 0.0
    top_b = float(b[seen_b].max())
    if mode in ('max', 'mean'):
        seen_n = inside & (n >= 0)
        if not seen_n.any():
            return 0.0
        if mode == 'mean':
            return float(b[seen_b].mean()) - float(n[seen_n].mean())
        return top_b - float(n[seen_n].max())
    cells = seen_b & (b >= top_b - band) & (n >= 0)
    if not cells.any():
        return 0.0
    drops = b[cells] - n[cells]
    if mode == 'frac':
        return float(np.mean(drops >= 0.6 - 1e-6))
    return float(np.median(drops))
