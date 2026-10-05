import math, sys, collections
import numpy as np, yaml
from scipy import ndimage
import deforestation_monitoring.tree_detection as T
from deforestation_monitoring.visibility import classify_tree, removal_tier
ROOT = '/home/alig/rs1_18_ws/src/RS1_18'
g = np.load(sys.argv[1]); world = sys.argv[2]; half = float(sys.argv[3])
RES, OX, OY = 0.25, -40.0, -40.0
sc = g >= 0; chm = np.where(sc, g / 10.0, 0).astype(np.float32); grid = np.where(sc, g / 10.0, -1.0)
p = T.DetectionParams.from_dict(yaml.safe_load(open(f'{ROOT}/deforestation_monitoring/config/tree_detection_params.yaml'))['/**']['ros__parameters'])
p.chm_resolution = RES
truth = T.parse_tree_truth(f'{ROOT}/41068_ignition_bringup/worlds/{world}.sdf')
inbox = [t for t in truth if abs(t[1]) <= half - 2 and abs(t[2]) <= half - 2]
tier = {t[0]: removal_tier(t, classify_tree(grid, RES, (OX, OY), t)[0], truth) for t in inbox}
cs = T.smooth_chm(chm, sc, p.smooth_sigma / RES)
cs = np.where(sc, ndimage.grey_closing(cs, size=p.pit_fill_size), 0).astype(np.float32)
canopy, _ = T.canopy_mask_from_chm(cs, sc, p); canopy = T.refine_canopy_mask(canopy, p)
narrow = T.smooth_chm(chm, sc, p.narrow_sigma / RES)
# broad candidates before NMS
heights = np.where(canopy, cs, 0.0)
wm = np.clip(p.window_scale * heights + p.window_offset, p.min_window, p.max_window)
rpx = np.clip(np.ceil(wm / RES).astype(int), 1, 32)
cand = np.zeros(chm.shape, bool)
for s in np.unique(rpx):
    lm = ndimage.maximum_filter(heights, size=int(2*s+1), mode='constant'); cand |= (rpx == s) & (heights == lm)
cand &= canopy
cidx = np.argwhere(cand)
markers = T.treetop_markers(cs, canopy, p, raw_chm=chm, narrow_chm=narrow)
fh = chm[markers[:, 0], markers[:, 1]]
order = np.argsort(-fh); own = {}
for lid, i in enumerate(order, start=1): own[tuple(markers[i])] = lid
labels, _ = T.segment_crowns(cs, canopy, markers, p, flood_chm=chm)
dets_band = T.trees_from_labels(labels, cs, p, OX, OY, position_mode='band')
owner = {}; T.greedy_overlap_select(dets_band, p.overlap_threshold, owner=owner)
final = T.detect_trees(chm, sc, p, OX, OY)
pairs = sorted((math.hypot(d.x - t[1], d.y - t[2]), ti, di) for ti, t in enumerate(inbox) for di, d in enumerate(final))
ut, ud, mt = set(), set(), {}
for dd, ti, di in pairs:
    if dd > 1.5: break
    if ti in ut or di in ud: continue
    ut.add(ti); ud.add(di); mt[inbox[ti][0]] = dd
st = collections.defaultdict(collections.Counter)
for n, x, y in inbox:
    if tier[n] != 'canopy': continue
    sp = n.split('_')[0]
    ci = (int((x - OX) / RES), int((y - OY) / RES))
    if n in mt: st[sp]['detected'] += 1; continue
    mk = [m for m in markers if math.hypot(m[0]-ci[0], m[1]-ci[1]) * RES < 1.5]
    if not mk:
        c = [c for c in cidx if math.hypot(c[0]-ci[0], c[1]-ci[1]) * RES < 1.5]
        st[sp]['no marker: ' + ('no local max' if not c else 'NMS/saddle-merged')] += 1; continue
    m = min(mk, key=lambda m: math.hypot(m[0]-ci[0], m[1]-ci[1]))
    L = labels[m[0], m[1]]; mine = own[tuple(m)]
    if L != mine: st[sp]['marker stolen by another crown'] += 1; continue
    if mine in owner: st[sp]['overlap-suppressed (lobe-merged)'] += 1; continue
    d = [d for d in final if d.label == mine]
    if d:
        st[sp][f'crown off > 1.5 m ({math.hypot(d[0].x-x, d[0].y-y):.1f})'] += 1
    else:
        st[sp]['other'] += 1
for sp, c in st.items():
    print(sp, dict(c))
