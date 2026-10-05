import os
import math, collections
import numpy as np, yaml
from scipy import ndimage
from concurrent.futures import ProcessPoolExecutor
import deforestation_monitoring.tree_detection as T
from deforestation_monitoring.visibility import classify_tree, removal_tier
R = '/home/alig/rs1_18_ws/src/RS1_18/'
RES, O = 0.25, -40.0
truth = T.parse_tree_truth(R + '41068_ignition_bringup/worlds/dense_forest.sdf')
full = [t for t in truth if abs(t[1]) <= 30 and abs(t[2]) <= 30]
base = yaml.safe_load(open(R + 'deforestation_monitoring/config/tree_detection_params.yaml'))['/**']['ros__parameters']
frames = []
for f in ('dense_survey_None_8', 'removal4_270.0_8', 'removal6_270.0_8', 'removal8_270.0_8'):
    frames += list(np.load(f'/home/alig/rs1_18_ws/bags/.map_cache/{f}.npz')['maps'])
cell = lambda x, y: np.array([int((x - O) / RES), int((y - O) / RES)])
def run(g):
    p = T.DetectionParams.from_dict(base); p.chm_resolution = RES
    sc = g >= 0; chm = np.where(sc, g, 0).astype(np.float32)
    cs = T.smooth_chm(chm, sc, p.smooth_sigma / RES)
    cs = np.where(sc, ndimage.grey_closing(cs, size=p.pit_fill_size), 0).astype(np.float32)
    canopy, _ = T.canopy_mask_from_chm(cs, sc, p); canopy = T.refine_canopy_mask(canopy, p)
    h = np.where(canopy, cs, 0.0)
    wm = np.clip(p.window_scale * h + p.window_offset, p.min_window, p.max_window)
    rpx = np.clip(np.ceil(wm / RES).astype(int), 1, 32)
    cand = np.zeros(h.shape, bool)
    for s in np.unique(rpx):
        cand |= (rpx == s) & (h == ndimage.maximum_filter(h, size=int(2*s+1), mode='constant'))
    cand = np.argwhere(cand & canopy)
    # after NMS only (no narrow, no saddle merge)
    order = np.argsort(-h[cand[:, 0], cand[:, 1]]); nms = []
    for i in cand[order]:
        if all(math.hypot(i[0]-k[0], i[1]-k[1]) >= p.min_sep / RES for k in nms): nms.append(i)
    narrow = T.smooth_chm(chm, sc, p.narrow_sigma / RES)
    markers = T.treetop_markers(cs, canopy, p, raw_chm=chm, narrow_chm=narrow)
    dets = T.detect_trees(chm, sc, p, O, O)
    out = []
    for n, x, y in full:
        if not n.startswith('oak') or abs(x) > 28 or abs(y) > 28: continue
        if removal_tier((n, x, y), classify_tree(g, RES, (O, O), (n, x, y))[0], truth) != 'canopy': continue
        c = cell(x, y)
        d = min(dets, key=lambda q: math.hypot(q.x - x, q.y - y)); dd = math.hypot(d.x - x, d.y - y)
        # is that detection closer to another trunk?
        other = min((math.hypot(d.x - t[1], d.y - t[2]), t[0]) for t in full if t[0] != n)
        if dd <= 2.0 and dd <= other[0]: out.append((n, 'ok')); continue
        near = lambda arr, r: [m for m in arr if math.hypot(m[0]-c[0], m[1]-c[1]) * RES <= r]
        if not near(cand, 2.5): cause = 'no local max within 2.5 m'
        elif not near(nms, 2.5): cause = 'NMS: taller peak within min_sep'
        elif not near(markers, 2.5): cause = 'saddle-merged into a neighbour'
        elif dd <= 2.0: cause = 'det closer to another trunk'
        else: cause = f'marker but crown position {min(dd, 9):.0f}+ m off'
        out.append((n, cause))
    return out
with ProcessPoolExecutor(16) as ex:
    res = list(ex.map(run, frames))
per = collections.defaultdict(collections.Counter)
for fr in res:
    for n, c in fr: per[n][c] += 1
tot = collections.Counter()
unstable = 0
for n, c in per.items():
    if c['ok'] / sum(c.values()) < 0.9:
        unstable += 1
        for k, v in c.items():
            if k != 'ok': tot[k] += v
print(f'unstable canopy oaks (<90% at 2 m): {unstable}/{len(per)}; miss causes over their frames:')
for k, v in tot.most_common(): print(f'  {v:4d}  {k}')
import pickle
pickle.dump(dict(per), open(os.path.join(os.environ.get('EVAL_DIR', os.path.expanduser('~/rs1_18_ws/eval')), 'oak_per.pkl'), 'wb'))
by = {t[0]: t for t in truth}
rows = []
for n, c in per.items():
    x, y = by[n][1], by[n][2]
    ds = sorted(math.hypot(t[1]-x, t[2]-y) for t in truth if t[0] != n and t[0].startswith('oak'))
    anyd = sorted(math.hypot(t[1]-x, t[2]-y) for t in truth if t[0] != n)
    rows.append((c['ok'] / sum(c.values()), ds[0], ds[1], sum(d < 5.0 for d in ds), anyd[0], n))
rows.sort()
print('rate  nn_oak  2nd_oak  oaks<5m  nn_any  name')
for r in rows: print(f'{r[0]:.2f}  {r[1]:5.1f}  {r[2]:6.1f}  {r[3]:6d}  {r[4]:6.1f}  {r[5]}')
