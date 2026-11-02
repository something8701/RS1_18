# logging_evidence.py: On the ground: tree stumps, felled logs (and the waste bags)
import math
import random
from typing import Dict, List, Sequence

import numpy as np

from .waste import BAG_RADII

CLASSES = ['stump', 'log', 'bag']

BARK = [(0.33, 0.22, 0.13), (0.28, 0.20, 0.14), (0.40, 0.28, 0.18), (0.24, 0.17, 0.11)]
WOOD = [(0.86, 0.71, 0.48), (0.80, 0.63, 0.40), (0.90, 0.78, 0.58)]


def _mat(rgb) -> str:
    r, g, b = rgb
    return (f'<material><ambient>{r:.2f} {g:.2f} {b:.2f} 1</ambient>'
            f'<diffuse>{r:.2f} {g:.2f} {b:.2f} 1</diffuse>'
            f'<specular>0.05 0.05 0.05 1</specular></material>')


def random_params(kind: str, rng: random.Random = random) -> Dict:
    if kind == 'stump':
        r = rng.uniform(0.14, 0.32)
        return {'kind': kind, 'r': r, 'h': rng.uniform(0.25, 0.75),
                'bark': rng.choice(BARK), 'wood': rng.choice(WOOD),
                'roots': rng.randint(3, 5), 'root_yaw': rng.uniform(0, 2 * math.pi)}
    if kind == 'log':
        return {'kind': kind, 'r': rng.uniform(0.10, 0.24), 'length': rng.uniform(1.8, 5.0),
                'bark': rng.choice(BARK), 'wood': rng.choice(WOOD),
                'stubs': rng.randint(0, 3), 'stub_seed': rng.randint(0, 10_000)}
    if kind == 'bag':
        return {'kind': kind}
    raise ValueError(kind)


def stump_sdf(name: str, x: float, y: float, yaw: float, p: Dict, collision: bool = True) -> str:
    r, h = p['r'], p['h']
    parts = [f'<visual name="bark"><pose>0 0 {h / 2:.3f} 0 0 0</pose><geometry><cylinder>'
             f'<radius>{r:.3f}</radius><length>{h:.3f}</length></cylinder></geometry>'
             f'{_mat(p["bark"])}</visual>',
             f'<visual name="cut"><pose>0 0 {h + 0.008:.3f} 0 0 0</pose><geometry><cylinder>'
             f'<radius>{0.9 * r:.3f}</radius><length>0.016</length></cylinder></geometry>'
             f'{_mat(p["wood"])}</visual>']
    for k in range(p['roots']):
        a = p['root_yaw'] + 2 * math.pi * k / p['roots']
        parts.append(f'<visual name="root{k}"><pose>{0.85 * r * math.cos(a):.3f} '
                     f'{0.85 * r * math.sin(a):.3f} 0.04 0 0 {a:.3f}</pose><geometry><ellipsoid>'
                     f'<radii>{0.75 * r:.3f} {0.3 * r:.3f} {0.22 * r + 0.03:.3f}</radii>'
                     f'</ellipsoid></geometry>{_mat(p["bark"])}</visual>')
    if collision:
        parts.append(f'<collision name="c"><pose>0 0 {h / 2:.3f} 0 0 0</pose><geometry>'
                     f'<cylinder><radius>{r:.3f}</radius><length>{h:.3f}</length></cylinder>'
                     f'</geometry></collision>')
    return _model(name, x, y, 0.0, yaw, parts)


def _stub_list(p: Dict):
    rng = random.Random(p['stub_seed'])
    return [(rng.uniform(-0.35, 0.35) * p['length'], rng.uniform(0, 2 * math.pi),
             rng.uniform(0.15, 0.4)) for _ in range(p['stubs'])]


def log_sdf(name: str, x: float, y: float, yaw: float, p: Dict, collision: bool = True) -> str:
    r, L = p['r'], p['length']
    parts = [f'<visual name="bark"><pose>0 0 0 0 1.5708 0</pose><geometry><cylinder>'
             f'<radius>{r:.3f}</radius><length>{L:.3f}</length></cylinder></geometry>'
             f'{_mat(p["bark"])}</visual>']
    for s, name_end in ((1, 'end_a'), (-1, 'end_b')):
        parts.append(f'<visual name="{name_end}"><pose>{s * (L / 2 + 0.006):.3f} 0 0 0 1.5708 0'
                     f'</pose><geometry><cylinder><radius>{0.9 * r:.3f}</radius>'
                     f'<length>0.012</length></cylinder></geometry>{_mat(p["wood"])}</visual>')
    for k, (u, a, sl) in enumerate(_stub_list(p)):
        cy, cz = (r + sl / 2) * math.cos(a), (r + sl / 2) * math.sin(a)
        parts.append(f'<visual name="stub{k}"><pose>{u:.3f} {cy:.3f} {max(cz, -0.5 * r):.3f} '
                     f'{a - math.pi / 2:.3f} 0 0</pose><geometry><cylinder>'
                     f'<radius>{0.25 * r + 0.01:.3f}</radius><length>{sl:.3f}</length></cylinder>'
                     f'</geometry>{_mat(p["bark"])}</visual>')
    if collision:
        parts.append(f'<collision name="c"><pose>0 0 0 0 1.5708 0</pose><geometry><cylinder>'
                     f'<radius>{r:.3f}</radius><length>{L:.3f}</length></cylinder></geometry>'
                     f'</collision>')
    return _model(name, x, y, r, yaw, parts)


def _model(name: str, x: float, y: float, z: float, yaw: float, parts: List[str]) -> str:
    return (f'<sdf version="1.8"><model name="{name}"><static>true</static>'
            f'<pose>{x:.3f} {y:.3f} {z:.3f} 0 0 {yaw:.4f}</pose><link name="link">'
            + ''.join(parts) + '</link></model></sdf>')


def evidence_sdf(name: str, x: float, y: float, yaw: float, p: Dict) -> str:
    if p['kind'] == 'stump':
        return stump_sdf(name, x, y, yaw, p)
    if p['kind'] == 'log':
        return log_sdf(name, x, y, yaw, p)
    from .waste import bag_sdf
    return bag_sdf(name, x, y, yaw)


def model_z(p: Dict) -> float:
    if p['kind'] == 'log':
        return p['r']
    if p['kind'] == 'bag':
        return BAG_RADII[2]
    return 0.0


def _cylinder_points(r: float, h: float, n_around: int, n_up: int) -> np.ndarray:
    a = np.linspace(0, 2 * np.pi, n_around, endpoint=False)
    z = np.linspace(0, h, n_up)
    A, Z = np.meshgrid(a, z)
    side = np.stack([r * np.cos(A).ravel(), r * np.sin(A).ravel(), Z.ravel()], 1)
    rr = np.linspace(0, r, 4)
    R, A2 = np.meshgrid(rr, a)
    top = np.stack([(R * np.cos(A2)).ravel(), (R * np.sin(A2)).ravel(),
                    np.full(R.size, h)], 1)
    return np.vstack([side, top])


def surface_points(p: Dict, x: float, y: float, yaw: float) -> np.ndarray:
    if p['kind'] == 'stump':
        pts = [_cylinder_points(p['r'], p['h'] + 0.016, 24, 8)]
        for k in range(p['roots']):
            a = p['root_yaw'] + 2 * math.pi * k / p['roots']
            t = np.linspace(-1, 1, 7)
            pts.append(np.stack([0.85 * p['r'] * math.cos(a) + 0.75 * p['r'] * t * math.cos(a),
                                 0.85 * p['r'] * math.sin(a) + 0.75 * p['r'] * t * math.sin(a),
                                 np.full(t.size, 0.04)], 1))
        local = np.vstack(pts)
    elif p['kind'] == 'log':
        r, L = p['r'], p['length']
        c = _cylinder_points(r, L + 0.012, 20, 24)
        local = np.stack([c[:, 2] - (L + 0.012) / 2, c[:, 1], c[:, 0] + r], 1)
        bottom_cap = local.copy()
        bottom_cap[local[:, 0] > (L + 0.012) / 2 - 1e-6, 0] = -(L + 0.012) / 2
        local = np.vstack([local, bottom_cap])
    else:
        rx, ry, rz = BAG_RADII
        u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 20), np.linspace(-np.pi / 2, np.pi / 2, 10))
        local = np.stack([(rx * np.cos(v) * np.cos(u)).ravel(), (ry * np.cos(v) * np.sin(u)).ravel(),
                          (rz + rz * np.sin(v)).ravel()], 1)
    c, s = math.cos(yaw), math.sin(yaw)
    world = np.empty_like(local)
    world[:, 0] = x + c * local[:, 0] - s * local[:, 1]
    world[:, 1] = y + s * local[:, 0] + c * local[:, 1]
    world[:, 2] = local[:, 2]
    return world


def footprint_radius(p: Dict) -> float:
    if p['kind'] == 'stump':
        return 1.6 * p['r']
    if p['kind'] == 'log':
        return p['length'] / 2 + 0.3
    return BAG_RADII[0]


def place_ok(x: float, y: float, rad: float, trees: Sequence, placed: Sequence,
             trunk_clear: float = 0.6) -> bool:
    for _, tx, ty in trees:
        if math.hypot(x - tx, y - ty) < rad + trunk_clear:
            return False
    return all(math.hypot(x - px, y - py) > rad + pr + 0.3 for px, py, pr in placed)


def params_for(kind: str, name: str) -> Dict:
    p = random_params(kind, random.Random(f'{kind}:{name}'))
    if kind == 'stump':
        p['h'] = min(p['h'], 0.7)
    return p


def log_spot_near(tree, trees: Sequence, length: float,
                  avoid: Sequence = (), dists=(3.0, 4.0, 2.5)):
    name, tx, ty = tree
    start = (sum(map(ord, name)) * 53) % 360
    others = [(n, x, y) for n, x, y in trees if n != name]
    for dist in dists:
        for k in range(16):
            a = math.radians(start + 22.5 * k)
            x, y = tx + dist * math.cos(a), ty + dist * math.sin(a)
            yaw = a + math.pi / 2
            pts = [(x + s * math.cos(yaw), y + s * math.sin(yaw))
                   for s in np.linspace(-length / 2, length / 2, 9)]
            if all(math.hypot(px - ox, py - oy) > 0.9 for px, py in pts for _, ox, oy in others) \
                    and all(math.hypot(px - ax, py - ay) > 1.5 for px, py in pts for ax, ay in avoid):
                return round(x, 2), round(y, 2), round(yaw, 3)
    return None


def score_evidence(truth: Sequence[Dict], items: Sequence[Dict]) -> Dict:
    lim = {'stump': 1.5, 'log': 2.5, 'bag': 1.5}
    out = {}
    for kind in ('stump', 'log', 'bag'):
        tk = [t for t in truth if t['kind'] == kind]
        ik = [i for i in items if i['kind'] == kind]
        errs = []
        for t in tk:
            d = min((math.hypot(i['x'] - t['x'], i['y'] - t['y']) for i in ik), default=1e9)
            if d <= lim[kind]:
                errs.append(d)
        false = sum(1 for i in ik if all(math.hypot(i['x'] - t['x'], i['y'] - t['y'])
                                         > lim[kind] + 1.0 for t in tk))
        out[kind] = {'total': len(tk), 'found': len(errs), 'false': false,
                     'mean_err_m': round(float(np.mean(errs)), 2) if errs else None}
    return out
