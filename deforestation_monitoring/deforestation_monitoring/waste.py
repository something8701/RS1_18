# waste.py: Land waste (red garbage bags): placing them, seeing them, tracking them
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

RED_MIN = 90
RED_RATIO = 2.0

BAG_RADII = (0.40, 0.30, 0.22)
BAG_HEIGHT = 2 * BAG_RADII[2]


def red_mask(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    return (r >= RED_MIN) & (r >= RED_RATIO * g) & (r >= RED_RATIO * b)


def red_fraction_bgr(image: np.ndarray) -> float:
    if image is None or image.size == 0:
        return 0.0
    return float(red_mask(image[..., ::-1]).mean())


def choose_bag_positions(trees: Sequence[Tuple[str, float, float]], n_open: int = 4,
                         n_under: int = 2, reach: float = 22.0,
                         spacing: float = 8.0) -> List[Tuple[float, float, str]]:
    oaks = np.array([(x, y) for n, x, y in trees if n.startswith('oak')] or [(1e9, 1e9)])
    pines = np.array([(x, y) for n, x, y in trees if n.startswith('pine')] or [(1e9, 1e9)])
    cand = []
    for gx in np.arange(-reach, reach + 0.01, 1.0):
        for gy in np.arange(-reach, reach + 0.01, 1.0):
            if math.hypot(gx, gy) > reach or math.hypot(gx, gy) < 4.0:
                continue
            do = float(np.min(np.hypot(oaks[:, 0] - gx, oaks[:, 1] - gy)))
            dp = float(np.min(np.hypot(pines[:, 0] - gx, pines[:, 1] - gy)))
            if dp < 2.0:
                continue
            if do >= 6.5:
                cand.append((gx, gy, 'open', do))
            elif 2.0 <= do <= 3.0:
                cand.append((gx, gy, 'under', do))
    chosen: List[Tuple[float, float, str]] = []
    for kind, n in (('open', n_open), ('under', n_under)):
        pool = [c for c in cand if c[2] == kind]
        pool.sort(key=lambda c: (c[0], c[1]))
        for _ in range(n):
            best, best_d = None, -1.0
            for c in pool:
                d = min((math.hypot(c[0] - p[0], c[1] - p[1]) for p in chosen), default=99.0)
                if d >= spacing and d > best_d:
                    best, best_d = c, d
            if best is None:
                break
            chosen.append((float(best[0]), float(best[1]), kind))
    return chosen


def bag_sdf(name: str, x: float, y: float, yaw: float = 0.0) -> str:
    rx, ry, rz = BAG_RADII
    red = '<ambient>0.75 0.04 0.04 1</ambient><diffuse>0.85 0.06 0.06 1</diffuse>' \
          '<specular>0.3 0.3 0.3 1</specular>'
    return (f'<sdf version="1.8"><model name="{name}"><static>true</static>'
            f'<pose>{x} {y} {rz} 0 0 {yaw}</pose><link name="link">'
            f'<visual name="v"><geometry><ellipsoid><radii>{rx} {ry} {rz}</radii>'
            f'</ellipsoid></geometry><material>{red}</material></visual>'
            f'</link></model></sdf>')


def bag_spot_near(tree: Tuple[str, float, float], trees: Sequence[Tuple[str, float, float]],
                  near: Tuple[float, float] = (1.5, 3.0),
                  clear: float = 1.2) -> Optional[Tuple[float, float]]:
    name, tx, ty = tree
    start = (sum(map(ord, name)) * 37) % 360
    others = [(x, y) for n, x, y in trees if n != name]
    for dist in (near[0] + 0.5 * (near[1] - near[0]), near[1], near[0]):
        for k in range(12):
            a = math.radians(start + 30 * k)
            x, y = tx + dist * math.cos(a), ty + dist * math.sin(a)
            if all(math.hypot(x - ox, y - oy) >= clear for ox, oy in others):
                return round(x, 2), round(y, 2)
    return None


def spawn_sdf(world: str, name: str, sdf: str, timeout_s: float = 15.0) -> Tuple[bool, str]:
    import os
    import subprocess
    import tempfile
    fd, path = tempfile.mkstemp(suffix='.sdf', prefix=f'{name}_', text=True)
    with os.fdopen(fd, 'w') as fh:
        fh.write(sdf)
    cmd = ['ign', 'service', '-s', f'/world/{world}/create', '--timeout', '2000',
           '--reqtype', 'ignition.msgs.EntityFactory', '--reptype', 'ignition.msgs.Boolean',
           '--req', f'sdf_filename: "{path}" name: "{name}"']
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return False, str(exc)
    out = (r.stdout + r.stderr).strip()
    return r.returncode == 0 and 'error' not in out.lower() and 'false' not in out.lower(), out


@dataclass
class RedBlob:
    x: float
    y: float
    pixels: float
    area_m2: float


def find_red_blobs(red_hits: np.ndarray, res: float, origin_x: float, origin_y: float,
                   min_cell_hits: float = 2.0, min_pixels: float = 15.0,
                   max_area_m2: float = 6.0) -> List[RedBlob]:
    mask = red_hits >= min_cell_hits
    if not mask.any():
        return []
    labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
    blobs = []
    for i in range(1, n + 1):
        cells = labels == i
        total = float(red_hits[cells].sum())
        area = float(cells.sum()) * res * res
        if total < min_pixels or area > max_area_m2:
            continue
        ix, iy = np.nonzero(cells)
        w = red_hits[cells]
        blobs.append(RedBlob(x=origin_x + (float(np.average(ix, weights=w)) + 0.5) * res,
                             y=origin_y + (float(np.average(iy, weights=w)) + 0.5) * res,
                             pixels=total, area_m2=area))
    return blobs


DETECTED, CONFIRMED, NOT_FOUND = 0, 1, 2


@dataclass
class Site:
    id: int
    x: float
    y: float
    area_m2: float
    pixels: float
    status: int = DETECTED
    flagged: bool = False
    history: List[int] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        if self.status == CONFIRMED:
            return 100.0
        return min(95.0, 40.0 + self.pixels / 4.0)


class SiteTracker:

    def __init__(self, match_dist: float = 2.0):
        self.match_dist = match_dist
        self.sites: List[Site] = []

    def _nearest(self, x: float, y: float) -> Optional[Site]:
        best = min(self.sites, key=lambda s: math.hypot(s.x - x, s.y - y), default=None)
        if best is None or math.hypot(best.x - x, best.y - y) > self.match_dist:
            return None
        return best

    def update(self, blobs: Sequence[RedBlob]) -> List[Site]:
        new = []
        for b in blobs:
            s = self._nearest(b.x, b.y)
            if s is None:
                s = Site(id=len(self.sites) + 1, x=b.x, y=b.y, area_m2=b.area_m2, pixels=b.pixels)
                self.sites.append(s)
                new.append(s)
            else:
                s.x, s.y, s.area_m2, s.pixels = b.x, b.y, b.area_m2, max(s.pixels, b.pixels)
        return new

    def inspected(self, x: float, y: float, found: bool) -> Optional[Site]:
        s = self._nearest(x, y)
        if s is not None:
            s.status = CONFIRMED if found else NOT_FOUND
            s.history.append(s.status)
        return s
