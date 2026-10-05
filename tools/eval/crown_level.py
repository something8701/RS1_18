"""Secondary, crown-level view of removal-test reports (not the official
score). For each removed canopy tree: any LOST within FOUND_R m of its
trunk counts as found. Each LOST is then: the first report for a removed
tree, a duplicate (another LOST within CROWN_R of an already-found removed
trunk), or a false alarm (no removed trunk within CROWN_R).

usage: crown_level.py REPORT.json [...]
"""
import json
import math
import sys

FOUND_R, CROWN_R = 3.0, 5.5

tot = [0, 0, 0, 0]
for path in sys.argv[1:]:
    d = json.load(open(path))
    removed = [tuple(r) for r in d['removed']]
    tiers = {t['name']: t.get('tier') for t in d['result']['trees']}
    lost = [e for e in d['events'] if e['type'] == 'LOST' and not e.get('before_removal')]
    found, dup, false = set(), 0, 0
    for e in sorted(lost, key=lambda e: e.get('t', 0)):
        dists = sorted((math.hypot(e['x'] - x, e['y'] - y), n) for n, x, y in removed)
        near = [n for dd, n in dists if dd <= FOUND_R and n not in found]
        if near:
            found.add(near[0])
        elif dists and dists[0][0] <= CROWN_R:
            dup += 1
        else:
            false += 1
    canopy = [n for n, _, _ in removed if tiers.get(n) == 'canopy']
    cf = sum(1 for n in canopy if n in found)
    tot[0] += cf; tot[1] += len(canopy); tot[2] += dup; tot[3] += false
    name = path.split('/')[-1].replace('.report.json', '')
    print(f'{name:18s} canopy found (crown level) {cf}/{len(canopy)}  duplicates {dup}  '
          f'false alarms {false}  missed {[n for n in canopy if n not in found]}')
print(f'{"TOTAL":18s} canopy found {tot[0]}/{tot[1]}  duplicates {tot[2]}  false alarms {tot[3]}')
