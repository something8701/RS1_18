# radio_model.py: Radio links between the base station, the drone and the Husky (pure functions)
import math
from typing import Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float, float]


def fspl_db(distance: float, f_ghz: float) -> float:
    return 20.0 * math.log10(max(distance, 1.0)) + 20.0 * math.log10(f_ghz * 1000.0) - 27.55


def foliage_db(depth: float, f_ghz: float) -> float:
    if depth <= 0.0:
        return 0.0
    k = f_ghz ** 0.284
    if depth <= 14.0:
        return 0.45 * k * depth
    return 1.33 * k * min(depth, 400.0) ** 0.588


def foliage_depth(heights: Optional[np.ndarray], res: float, ox: float, oy: float,
                  a: Point, b: Point, step: float = 0.25) -> float:
    if heights is None:
        return 0.0
    length = math.dist(a, b)
    n = max(1, int(math.ceil(length / step)))
    t = (np.arange(n) + 0.5) / n
    x = a[0] + (b[0] - a[0]) * t
    y = a[1] + (b[1] - a[1]) * t
    z = a[2] + (b[2] - a[2]) * t
    ix = np.floor((x - ox) / res).astype(int)
    iy = np.floor((y - oy) / res).astype(int)
    h, w = heights.shape
    on = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    top = np.full(n, -1.0)
    top[on] = heights[iy[on], ix[on]]
    return float(np.count_nonzero(z < top)) * length / n


def link_margin(a: Point, b: Point, heights: Optional[np.ndarray], res: float,
                ox: float, oy: float, f_ghz: float, tx_dbm: float, gains_dbi: Sequence[float],
                sensitivity_dbm: float, fade_margin_db: float) -> Tuple[float, float, float]:
    d = math.dist(a, b)
    foliage = foliage_depth(heights, res, ox, oy, a, b)
    received = tx_dbm + sum(gains_dbi) - fspl_db(d, f_ghz) - foliage_db(foliage, f_ghz)
    return received - sensitivity_dbm - fade_margin_db, d, foliage
