#!/usr/bin/env python3
"""Detector-independent "visible from above" classification.

A tree counts as canopy (visible) when its own top is part of the canopy
surface the drone measured before anything changed:

    top = max canopy height within ``top_radius`` of the trunk
    visible     if  H_species - below <= top <= H_species + above
    understory  if  top >  H_species + above   (something taller covers it)
    unobserved  if  the cells were never scanned, or top < H_species - below

Only the simulator's trunk position, a per-species top height and the
pre-change canopy map are used, never the detector's output, so a test
cannot excuse a detector miss by calling the tree "invisible".

Species heights are measured on isolated trees (``species_top_heights``):
oak 6.2 m, pine 4.4-4.6 m for the Fuel models used here.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

Tree = Tuple[str, float, float]

# Measured top heights (m) of the Fuel tree models, isolated, raw LiDAR map.
DEFAULT_SPECIES_HEIGHT = {'oak': 6.2, 'pine': 4.6}


def species_of(name: str) -> str:
    return name.split('_')[0]


def disc_max(grid: np.ndarray, res: float, origin: Tuple[float, float],
             x: float, y: float, radius: float) -> Optional[float]:
    """Max of scanned cells (>= 0) within ``radius`` of (x, y); None if none.

    ``grid`` is indexed [ix, iy] in metres, -1 (or any negative) = unscanned.
    """
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
    """Median measured top per species (use on *isolated* trees only)."""
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
    """Return (``visible`` | ``understory`` | ``unobserved``, top height)."""
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


# A Fuel oak crown is clumps of branches reaching up to 5.2 m from the trunk
# (measured on real maps); each clump looks like a pine spire in a height
# map. A visible pine within this distance of an oak trunk shares the oak's
# crown footprint and cannot be separated from it by a height model alone.
CROWN_SHARE_RADIUS = 5.5


def removal_tier(tree: Tree, visibility_class: str, truth: Sequence[Tree],
                 share_radius: float = CROWN_SHARE_RADIUS) -> str:
    """``canopy`` | ``crown-shared`` | ``understory`` | ``unobserved``.

    Geometric and detector-independent: visibility class from the
    pre-change canopy map, plus the SDF distance from a pine to the
    nearest oak trunk.
    """
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
    """How far a tree's top fell between two CHMs ([ix, iy], -1 unscanned).

    ``mode='max'``: disc max before minus disc max now. One cell that still
    reads high (a neighbour's overhanging branch, a cell not re-read since)
    can hide a removal.

    ``mode='cells'``: the cells that formed the top at ``before`` (within
    ``radius``, height >= disc max - ``band``) are re-read in ``now``; the
    result is the median of their drops. A neighbour's lower overhang is
    not a top cell, so it cannot mask the loss. Returns 0 if unknown.

    ``mode='frac'``: the fraction of those top cells that fell >= 0.6 m.

    ``mode='mean'``: disc mean before minus disc mean now (scanned cells).
    A few cells a neighbour's branch holds high barely move it, while a cut
    crown takes the whole disc down.
    """
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
    if mode == 'frac':          # fraction of top cells that fell >= 0.6 m
        return float(np.mean(drops >= 0.6 - 1e-6))   # 0.1 m map steps
    return float(np.median(drops))
