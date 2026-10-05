"""Sweep LOST-gate persistence rules over live dumps and replay measurements.

Rule: top frac >= 0.9 & max drop >= 0.3 & ground+drop >= 6 [& nearest
detection >= ND m], held continuously for >= T seconds (and >= 3 ticks).
Live dumps come from live_gate_eval.py --dump; replays from offline_replay
runs with alerts disabled (lost_gate rows). EVAL_DIR as in the other scripts.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from gate_sim import pairing, W  # noqa: E402
from deforestation_monitoring.tree_detection import parse_tree_truth  # noqa: E402

EVAL = os.environ.get('EVAL_DIR', os.path.expanduser('~/rs1_18_ws/eval'))
LIVE = ['removal9_canopy', 'removal10_mixed', 'removal11_canopy', 'removal12_canopy']
REPLAY = ['r4_m3', 'r6_m3', 'r8_m3', 'r9c_m3', 'sparse1_m3', 'show2_m3',
          'demo4_m3', 'densesurvey_m3']


def fires(rows, cond, t_min, gap):
    start, last, n = None, None, 0
    for v in rows:
        if last is not None and v['t'] - last > gap:
            start, n = None, 0
        last = v['t']
        if cond(v):
            if start is None:
                start, n = v['t'], 0
            n += 1
            if n >= 3 and v['t'] - start >= t_min:
                return True
        else:
            start, n = None, 0
    return False


def load():
    runs = []
    for r in LIVE:
        p = os.path.join(EVAL, 'gatedump', r + '_gate.json')
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        runs.append((r, 4.5, d['rows'], d['trunk'], set(d['names'])))
    for r in REPLAY:
        p = os.path.join(EVAL, 'replay', r + '.json')
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        trunk = {str(k): v for k, v in pairing(
            d['baseline'], parse_tree_truth(W + d['world'] + '.sdf')).items()}
        rows = {bid: [dict(t=x[0], gd=x[2] + x[3], max=x[9], frac=x[12],
                           nd=x[13] if len(x) > 13 else 99.0)
                      for x in g['rows']] for bid, g in d['lost_gate'].items()}
        runs.append((r, 1.5, rows, trunk, {x[0] for x in d.get('removed', [])}))
    return runs


def main():
    runs = load()
    print('runs:', ', '.join(r[0] for r in runs))
    grid = [(nd, t, 6) for nd in (0.0, 1.5, 2.0, 2.5) for t in (0, 10, 20, 30, 45)]
    if '--cells' in sys.argv:
        grid = [(0.0, 0, g) for g in (6, 5, 4, 3, 2)]
    for nd, t_min, cells in grid:
        if True:
            def cond(v, nd=nd, cells=cells):
                return (v['frac'] >= 0.9 - 1e-6 and v['max'] >= 0.3 - 1e-6
                        and v['gd'] >= cells and v['nd'] >= nd)
            caught = total = 0
            false, missed = [], []
            for name, gap, rows, trunk, removed in runs:
                for bid, rs in rows.items():
                    tn = trunk.get(bid)
                    fired = fires(rs, cond, t_min, gap)
                    if tn in removed:
                        total += 1
                        caught += fired
                        if not fired and tn != 'pine_111':
                            missed.append(f'{tn}@{name}')
                    elif fired:
                        false.append(f'{tn or "#" + bid}@{name}')
            print(f'cells>={cells} nd>={nd:3.1f} T>={t_min:2d}s: caught {caught}/{total}  '
                  f'false {len(false)} {false}  missed (excl. pine_111) {missed}')


if __name__ == '__main__':
    main()
