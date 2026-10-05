"""Simulate tracker LOST-gate rules on measurement replays (alerts disabled).

rows per unmatched tick: (t, count, g_raw, drop_raw, g_clu, g_comp, b_clu,
b_comp, top075, top100).  A tree is LOST when the rule holds on
`streak` consecutive unmatched ticks (1 s apart).
"""
import json
import os
import math
import sys

from deforestation_monitoring.tree_detection import parse_tree_truth

S = os.path.join(os.environ.get('EVAL_DIR', os.path.expanduser('~/rs1_18_ws/eval')), 'replay')
W = '/home/alig/rs1_18_ws/src/RS1_18/41068_ignition_bringup/worlds/'
RUNS = [a for a in sys.argv[1:] if not a.startswith('-')] or [
    'r4_meas', 'r6_meas', 'r8_meas']

F = dict(t=0, g=2, d=3, gclu=4, gcomp=5, bclu=6, bcomp=7, top75=8, top1=9, c75=10, c1=11, frac=12, nd=13)


def R(fn, name):
    fn.name = name
    return fn


def row(r):
    v = {k: (r[i] if i < len(r) else 0.0) for k, i in F.items()} | {'b': r[2] + r[3]}
    if len(r) <= F['nd']:
        v['nd'] = 99.0
    return v


RULES = [
    R(lambda v: v['g'] >= 10, 'ground>=10 (old)'),
    R(lambda v: v['top1'] >= 0.6 and v['b'] >= 6, 'v2: max top1>=0.6 & g+d>=6'),
    R(lambda v: v['c1'] >= 0.6 and v['b'] >= 6, 'cells top1>=0.6 & g+d>=6'),
    R(lambda v: (v['top1'] >= 0.6 and v['b'] >= 6) or (v['c1'] >= 0.6 and v['b'] >= 6 and v['nd'] >= 1.5), 'max OR (cells & nd>=1.5)'),
    R(lambda v: (v['top1'] >= 0.6 and v['b'] >= 6) or (v['c1'] >= 0.6 and v['b'] >= 6 and v['nd'] >= 2.0), 'max OR (cells & nd>=2.0)'),
    R(lambda v: (v['top1'] >= 0.6 and v['b'] >= 6) or (v['c1'] >= 0.6 and v['b'] >= 6 and v['nd'] >= 2.5), 'max OR (cells & nd>=2.5)'),
    R(lambda v: v['c1'] >= 0.6 and v['b'] >= 6 and v['nd'] >= 2.0, 'cells & nd>=2.0'),
    R(lambda v: v['frac'] >= 0.7 and v['b'] >= 6, 'frac>=0.7 & g+d>=6'),
    R(lambda v: v['frac'] >= 0.8 and v['b'] >= 6, 'frac>=0.8 & g+d>=6'),
    R(lambda v: v['frac'] >= 0.9 and v['b'] >= 6, 'frac>=0.9 & g+d>=6'),
    R(lambda v: v['c1'] >= 0.8 and v['b'] >= 6, 'cells top1>=0.8 & g+d>=6'),
    R(lambda v: v['c1'] >= 0.6 and v['b'] >= 3, 'cells top1>=0.6 & g+d>=3'),
    R(lambda v: v['c1'] >= 0.6, 'cells top1>=0.6 alone'),
    R(lambda v: v['c75'] >= 0.6 and v['b'] >= 6, 'cells top.75>=0.6 & g+d>=6'),
    R(lambda v: v['gclu'] >= 10, 'ground clustered>=10'),
    R(lambda v: v['gcomp'] >= 8, 'ground patch>=8'),
    R(lambda v: v['top1'] >= 0.8, 'top1.0 drop>=0.8'),
    R(lambda v: v['top1'] >= 1.0, 'top1.0 drop>=1.0'),
    R(lambda v: v['top75'] >= 0.8, 'top0.75 drop>=0.8'),
    R(lambda v: v['top75'] >= 1.0, 'top0.75 drop>=1.0'),
    R(lambda v: v['top1'] >= 0.8 and v['b'] >= 10, 'top1>=0.8 & ground+drop>=10'),
    R(lambda v: v['top1'] >= 0.8 and v['g'] + v['d'] >= 5, 'top1>=0.8 & ground+drop>=5'),
    R(lambda v: v['top75'] >= 0.8 and v['b'] >= 5, 'top.75>=0.8 & ground+drop>=5'),
    R(lambda v: v['g'] >= 10 or (v['top1'] >= 0.8 and v['b'] >= 10),
      'ground>=10 OR (top1>=0.8 & g+d>=10)'),
    R(lambda v: v['gclu'] >= 10 or (v['top1'] >= 0.8 and v['b'] >= 10),
      'gclu>=10 OR (top1>=0.8 & g+d>=10)'),
    R(lambda v: v['top1'] >= 0.8 and (v['gclu'] >= 5 or v['bclu'] >= 10),
      'top1>=0.8 & (gclu>=5 | bclu>=10)'),
]


def pairing(baseline, truth, radius=2.0):
    cands = sorted((math.hypot(b[1] - t[1], b[2] - t[2]), b[0], t[0])
                   for b in baseline for t in truth
                   if math.hypot(b[1] - t[1], b[2] - t[2]) <= radius)
    trunk_of, used = {}, set()
    for d, bid, tn in cands:
        if bid not in trunk_of and tn not in used:
            trunk_of[bid] = tn
            used.add(tn)
    return trunk_of


def first_lost(rows, rule, streak=3):
    run, last_t = 0, None
    for r in rows:
        v = row(r)
        if last_t is not None and v['t'] - last_t > 1.5:
            run = 0
        last_t = v['t']
        run = run + 1 if rule(v) else 0
        if run >= streak:
            return v['t']
    return None


data = {}
for run in RUNS:
    d = json.load(open(f'{S}/{run}.json'))
    truth = parse_tree_truth(W + d['world'] + '.sdf')
    trunk_of = pairing(d['baseline'], truth)
    removed = {r[0] for r in d.get('removed', [])}
    tiers = {t['name']: t.get('tier') for t in d.get('result', {}).get('trees', [])}
    data[run] = (d, trunk_of, removed, tiers)

print(f"{'rule':42s} " + ' | '.join(f'{r:>22s}' for r in RUNS))
for rule in RULES:
    cells = []
    for run in RUNS:
        d, trunk_of, removed, tiers = data[run]
        gate = {int(k): v for k, v in d['lost_gate'].items()}
        hit, miss, fs, fp = [], [], [], []
        for bid, g in gate.items():
            t = first_lost(g['rows'], rule)
            tn = trunk_of.get(bid)
            if tn in removed:
                (hit if t is not None else miss).append(tn)
            elif t is not None:
                (fs if tn is not None else fp).append(tn or f'#{bid}')
        canopy = [n for n in removed if tiers.get(n) == 'canopy']
        base_canopy = [n for n in canopy if n in trunk_of.values()]
        ok = [n for n in hit if tiers.get(n) == 'canopy']
        cells.append(f'{len(ok)}/{len(base_canopy)} bl, F {len(fs)}+{len(fp)}ph')
        if '-v' in sys.argv:
            cells[-1] += f' miss={[m for m in miss if tiers.get(m)=="canopy"]} fs={fs}'
    print(f'{rule.name:42s} ' + ' | '.join(f'{c:>22s}' for c in cells))
