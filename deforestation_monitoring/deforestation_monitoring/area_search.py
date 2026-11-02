# area_search.py: Husky area search (pure geometry): how big a suspected area is, and the path that sweeps it
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

Box = Tuple[float, float, float, float]


def loss_box(grid: np.ndarray, res: float, ox: float, oy: float, x: float, y: float,
             seed: float = 1.5, reach: float = 8.0, min_side: float = 4.0) -> Box:
    h, w = grid.shape
    r = int(math.ceil(reach / res))
    cx, cy = int((x - ox) / res), int((y - oy) / res)
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    box = [x, y, x, y]
    if x0 < x1 and y0 < y1:
        labels, n = ndimage.label(grid[y0:y1, x0:x1] < 0, structure=np.ones((3, 3)))
        if n:
            iy, ix = np.mgrid[y0:y1, x0:x1]
            d = np.hypot(ox + (ix + 0.5) * res - x, oy + (iy + 0.5) * res - y)
            near = np.unique(labels[(labels > 0) & (d <= seed)])
            if len(near):
                rows, cols = np.nonzero(np.isin(labels, near))
                box = [ox + (x0 + cols.min()) * res, oy + (y0 + rows.min()) * res,
                       ox + (x0 + cols.max() + 1) * res, oy + (y0 + rows.max() + 1) * res]
    for a, b in ((0, 2), (1, 3)):
        if box[b] - box[a] < min_side:
            c = (box[a] + box[b]) / 2
            box[a], box[b] = c - min_side / 2, c + min_side / 2
    return tuple(float(v) for v in box)


def sweep_path(box: Box, start: Tuple[float, float], margin: float = 2.0,
               spacing: float = 3.0) -> List[Tuple[float, float, float]]:
    xmin, ymin = box[0] - margin, box[1] - margin
    xmax, ymax = box[2] + margin, box[3] + margin
    along_x = (xmax - xmin) >= (ymax - ymin)
    u0, u1, v0, v1 = (xmin, xmax, ymin, ymax) if along_x else (ymin, ymax, xmin, xmax)
    su, sv = (start[0], start[1]) if along_x else (start[1], start[0])
    n = max(2, math.ceil((v1 - v0) / spacing) + 1)
    vs = [v0 + (v1 - v0) * k / (n - 1) for k in range(n)]
    if abs(sv - v1) < abs(sv - v0):
        vs.reverse()
    forward = abs(su - u0) <= abs(su - u1)
    out = []
    for v in vs:
        a, b = (u0, u1) if forward else (u1, u0)
        for u in (a, b):
            x, y = (u, v) if along_x else (v, u)
            dx, dy = ((b - a), 0.0) if along_x else (0.0, (b - a))
            out.append((x, y, math.atan2(dy, dx)))
        forward = not forward
    return out


def free_spot(cost: Optional[np.ndarray], res: float, ox: float, oy: float,
              x: float, y: float, max_shift: float = 1.5,
              free_below: int = 70) -> Optional[Tuple[float, float]]:
    if cost is None:
        return x, y
    h, w = cost.shape

    def ok(px, py):
        ix, iy = int(math.floor((px - ox) / res)), int(math.floor((py - oy) / res))
        return not (0 <= ix < w and 0 <= iy < h) or cost[iy, ix] < free_below

    if ok(x, y):
        return x, y
    for r in np.arange(0.25, max_shift + 1e-6, 0.25):
        for k in range(16):
            a = 2 * math.pi * k / 16
            px, py = x + r * math.cos(a), y + r * math.sin(a)
            if ok(px, py):
                return float(px), float(py)
    return None


def inside(box: Sequence[float], x: float, y: float, margin: float = 0.0) -> bool:
    return box[0] - margin <= x <= box[2] + margin and box[1] - margin <= y <= box[3] + margin


def box_overlap(a: Sequence[float], b: Sequence[float]) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    area = (a[2] - a[0]) * (a[3] - a[1])
    if w <= 0 or h <= 0 or area <= 0:
        return 0.0
    return w * h / area
