# tree_refine.py: Ground check of the drone's trees with the Husky's LiDAR (pure logic)
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

UNCHECKED, STANDING, NO_TRUNK, CUT_CONFIRMED, FALSE_LOST, HIDDEN = range(6)
NAMES = ['unchecked', 'standing', 'no trunk', 'cut confirmed', 'false LOST', 'hidden']


@dataclass
class Trunk:
    x: float
    y: float
    hits: int = 1
    last_seen: float = 0.0
    cleared: int = 0
    spread2: float = 0.0

    @property
    def spread(self) -> float:
        return math.sqrt(self.spread2)


class TrunkTracker:

    def __init__(self, merge: float = 0.6, forget: int = 8):
        self.merge = merge
        self.forget = forget
        self.trunks: List[Trunk] = []

    def add(self, centres: Sequence[Tuple[float, float]], t: float = 0.0):
        for x, y in centres:
            best = min(self.trunks, key=lambda k: math.hypot(k.x - x, k.y - y), default=None)
            d = math.hypot(best.x - x, best.y - y) if best is not None else math.inf
            if d <= self.merge:
                n = min(best.hits, 50)
                best.x = (best.x * n + x) / (n + 1)
                best.y = (best.y * n + y) / (n + 1)
                best.spread2 = (best.spread2 * n + d * d) / (n + 1)
                best.hits += 1
                best.last_seen, best.cleared = t, 0
            else:
                self.trunks.append(Trunk(x, y, last_seen=t))

    def seen_through(self, clear: np.ndarray):
        for k, c in zip(self.trunks, clear):
            if c:
                k.cleared += 1
        self.trunks = [k for k in self.trunks if k.cleared < self.forget]

    def confirmed(self, min_hits: int) -> List[Trunk]:
        return [t for t in self.trunks if t.hits >= min_hits]


def clear_views(sensor_xy: Tuple[float, float], sensor_yaw: float, angle_min: float,
                angle_increment: float, ranges: np.ndarray, spots: np.ndarray,
                max_dist: float = 12.0, margin: float = 0.6) -> np.ndarray:
    out = np.zeros(len(spots), bool)
    if len(spots) == 0:
        return out
    d = spots - np.asarray(sensor_xy)
    dist = np.hypot(d[:, 0], d[:, 1])
    bearing = np.arctan2(d[:, 1], d[:, 0]) - sensor_yaw
    idx = np.round(((bearing - angle_min) % (2 * math.pi)) / angle_increment).astype(int)
    ok = (dist <= max_dist) & (dist > 1.0) & (idx >= 0) & (idx < len(ranges))
    r = np.where(ok, ranges[np.clip(idx, 0, len(ranges) - 1)], 0.0)
    r = np.where(np.isfinite(r), r, np.inf)
    return ok & (r >= dist + margin)


@dataclass
class Refined:
    id: int
    drone_xy: Tuple[float, float]
    status: int = UNCHECKED
    ground_xy: Optional[Tuple[float, float]] = None
    trunk_hits: int = 0
    views: int = 0


@dataclass
class RefineConfig:
    match_radius: float = 1.5
    min_hits: int = 4
    min_views: int = 12
    hidden_clear: float = 3.0
    hidden_hits: int = 10
    hidden_spread: float = 0.3


def refine(drone_trees: Sequence[Tuple[int, float, float]], lost: Dict[int, float],
           trunks: Sequence[Trunk], views: Dict[int, int],
           cfg: RefineConfig = RefineConfig()) -> List[Refined]:
    good = [t for t in trunks if t.hits >= cfg.min_hits]
    out = [Refined(i, (x, y), views=views.get(i, 0)) for i, x, y in drone_trees]
    pair: Dict[int, Trunk] = {}
    if out and good:
        cost = np.array([[math.hypot(r.drone_xy[0] - t.x, r.drone_xy[1] - t.y) for t in good]
                         for r in out])
        big = cost.max() + 1e3
        rows, cols = linear_sum_assignment(np.where(cost <= cfg.match_radius, cost, big))
        for i, j in zip(rows, cols):
            if cost[i, j] <= cfg.match_radius:
                pair[i] = good[j]
    used = set(id(t) for t in pair.values())
    for i, r in enumerate(out):
        t = pair.get(i)
        is_lost = r.id in lost
        if t is not None and is_lost and t.last_seen <= lost[r.id]:
            t = None
        if t is not None:
            r.ground_xy, r.trunk_hits = (t.x, t.y), t.hits
            r.status = FALSE_LOST if is_lost else STANDING
        elif r.views >= cfg.min_views:
            r.status = CUT_CONFIRMED if is_lost else NO_TRUNK
    xs, ys = [x for _, x, _ in drone_trees], [y for _, _, y in drone_trees]
    inside = (lambda x, y: min(xs) - 1 <= x <= max(xs) + 1 and min(ys) - 1 <= y <= max(ys) + 1) \
        if drone_trees else None
    for t in good:
        if id(t) in used or t.hits < cfg.hidden_hits or t.spread > cfg.hidden_spread:
            continue
        if not inside:
            continue
        if all(math.hypot(r.drone_xy[0] - t.x, r.drone_xy[1] - t.y) > cfg.hidden_clear
               for r in out[:len(drone_trees)]) and inside(t.x, t.y):
            out.append(Refined(0, (t.x, t.y), HIDDEN, (t.x, t.y), t.hits))
    return out


def summary(refined: Sequence[Refined]) -> Dict[str, int]:
    counts = {n: 0 for n in NAMES}
    for r in refined:
        counts[NAMES[r.status]] += 1
    return counts
