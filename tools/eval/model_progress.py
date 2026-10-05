"""How the dense-forest results moved: one figure from the live removal-test reports.

usage: model_progress.py --out model_progress.png [--extra REPORT.md LABEL]

Every live dense-forest removal test with the default target set cuts the
same 10 trees (pine_75, pine_111, oak_78, oak_137, pine_117, oak_60, pine_53,
oak_153, oak_37, pine_163). Left: per run, the cut trees reported LOST as a
tree and those flagged by a canopy-loss alert (report rows "LOST ✓" and
"AREA ALERT ✓"; "not observable" is not counted as reported), with the test
verdict and the false reports. Right: the first run against the latest.
Reports are read from test_evidence/; the replay pass rates are set below.
"""
import argparse
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

EVIDENCE = os.path.join(os.path.dirname(__file__), '..', '..', 'test_evidence')
RUNS = [   # report, x label (date, what changed)
    ('removal_test_8_dense.md', 'run 8\n27 Sep\nheight map'),
    ('removal_test_17_dense_mixed.md', 'run 17\n29 Sep\nC + D'),
    ('removal_test_20_dense_mixed_camera.md', 'run 20\n+ camera\npine rule'),
    ('removal_test_23_dense_mixed_defaults.md', 'run 23\n+ area\nalerts'),
    ('removal_test_24_dense_mixed_attach.md', 'run 24\n+ off-crown\npines'),
    ('removal_test_25_dense_mixed_PASS.md', 'run 25\n+ pine\nadmission'),
    ('removal_test_26_dense_mixed_PASS.md', 'run 26\n+ merge\nlook-back'),
    ('removal_test_28_dense_mixed_camera_only.md', 'run 28\n30 Sep\n+ camera\nonly pines'),
    ('removal_test_29_dense_mixed_PASS_camera_only.md', 'run 29\nsame'),
    ('removal_test_30_dense_mixed_PASS_final.md', 'run 30\n+ median\n+ clump hold'),
]
REPLAY_START = (0, 13)      # dense recordings passing in replay, start of option A
REPLAY_NOW = (10, 10)       # after the 2026-09-30 fixes

SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
TREE_C, ALERT_C = '#2a78d6', '#eb6834'          # categorical slots 1 and 2
GOOD, BAD = '#0ca30c', '#d03b3b'                # status: good / critical


def parse(path):
    txt = open(path, encoding='utf-8').read()
    rows = re.findall(r'^\| (\w+_\d+) .*$', txt, re.M)
    lost = len(re.findall(r'^\| \w+_\d+ .*\| LOST ✓ \|', txt, re.M))
    alert = len(re.findall(r'^\| \w+_\d+ .*\| AREA ALERT ✓ \|', txt, re.M))
    wrong = int(re.search(r'\*\*Wrong:\*\* (\d+)', txt).group(1))
    verdict = 'PASS' if txt.startswith('# Removal test — PASS') else 'FAIL'
    tp = re.search(r'TP (\d+), FP (\d+), FN (\d+)', txt)
    return {'cut': len(rows), 'lost': lost, 'alert': alert, 'false': wrong,
            'verdict': verdict, 'tp': int(tp.group(1)) if tp else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--extra', nargs=2, action='append', default=[],
                    metavar=('REPORT', 'LABEL'), help='a later run to add at the end')
    a = ap.parse_args()
    runs = [(os.path.join(EVIDENCE, f), label) for f, label in RUNS] + \
           [(p, label.replace('\\n', '\n')) for p, label in a.extra]
    data = [(label, parse(p)) for p, label in runs]

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': INK,
                         'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2})
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=SURFACE)
    fig.text(0.035, 0.94, 'Dense forest: finding the cut trees, model by model',
             fontsize=24, weight='bold', color=INK)
    fig.text(0.035, 0.905, 'Live removal tests in the simulator, the same 10 trees cut every '
             'time, scored against the simulator ground truth.', fontsize=14, color=INK2)

    ax = fig.add_axes([0.05, 0.17, 0.60, 0.66], facecolor=SURFACE)
    xs = range(len(data))
    lost = [d['lost'] for _, d in data]
    alert = [d['alert'] for _, d in data]
    ax.bar(xs, lost, width=0.62, color=TREE_C, edgecolor=SURFACE, linewidth=2,
           label='reported LOST as a tree', zorder=3)
    ax.bar(xs, alert, width=0.62, bottom=lost, color=ALERT_C, edgecolor=SURFACE, linewidth=2,
           label='flagged by a canopy-loss alert', zorder=3)
    for x, (_, d) in zip(xs, data):
        top = d['lost'] + d['alert']
        ok = d['verdict'] == 'PASS'
        ax.text(x, top + 0.25, f"{top}/{d['cut']}", ha='center', va='bottom',
                fontsize=13, weight='bold', color=INK)
        ax.text(x, top + 0.95, ('✓ PASS' if ok else '✗ FAIL'), ha='center', va='bottom',
                fontsize=11, weight='bold', color=GOOD if ok else BAD)
        ax.text(x, top + 1.55, f"{d['false']} false", ha='center', va='bottom',
                fontsize=10, color=INK2)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([label for label, _ in data], fontsize=10)
    ax.set_ylim(0, 12.6)
    ax.set_yticks(range(0, 11, 2))
    ax.set_ylabel('cut trees reported (of 10)', fontsize=12)
    ax.grid(axis='y', color=GRID, lw=0.8, zorder=0)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(length=0)
    ax.legend(loc='upper left', fontsize=11, frameon=False, ncol=2,
              bbox_to_anchor=(0.0, 1.07))

    first, last = data[0][1], data[-1][1]
    tiles = [
        ('Cut trees reported', f"{first['lost'] + first['alert']} / 10",
         f"{last['lost'] + last['alert']} / 10"),
        ('False reports', f"{first['false']}", f"{last['false']}"),
        ('Real trees in the baseline\n(of 134 in the survey area)',
         f"{first['tp']}", f"{last['tp']}"),
        ('Recorded dense runs that\npass when replayed',
         f'{REPLAY_START[0]} of {REPLAY_START[1]}', f'{REPLAY_NOW[0]} of {REPLAY_NOW[1]}'),
    ]
    fig.text(0.70, 0.83, 'Before (27 Sep)', fontsize=13, color=INK2, weight='bold')
    fig.text(0.86, 0.83, 'Now', fontsize=13, color=INK2, weight='bold')
    y = 0.74
    for name, before, now in tiles:
        fig.text(0.70, y + 0.045, name, fontsize=12, color=INK2, va='bottom')
        fig.text(0.70, y, before, fontsize=30, color=INK2, va='center')
        fig.text(0.815, y, '→', fontsize=24, color=INK2, va='center')
        fig.text(0.86, y, now, fontsize=30, color=INK, weight='bold', va='center')
        y -= 0.165
    fig.text(0.70, 0.08, 'What changed: the drone camera. It tells pines from oaks by colour,\n'
             'sees through the canopy where a pine was cut, and finds pines the\n'
             'height map merges into oak crowns. Replay rate before: runs recorded\n'
             'without the camera; now: every dense recording with the camera.',
             fontsize=11, color=INK2, va='bottom')
    fig.savefig(a.out, facecolor=SURFACE)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
