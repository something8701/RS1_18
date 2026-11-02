#!/usr/bin/env python3
# drop_ghost_frames.py: DATASET [--apply]: set aside frames that may show an unlabelled leftover object
import json
import math
import os
import shutil
import sys
from collections import defaultdict

ds = sys.argv[1]
apply = '--apply' in sys.argv
HALF_FOV = math.radians(62)
rows = [json.loads(line) for line in open(f'{ds}/meta.jsonl')]
by_run = defaultdict(list)
for r in rows:
    by_run[r['stem'].rsplit('_', 2)[0]].append(r)
drop, total = [], 0
for run, rs in by_run.items():
    rs.sort(key=lambda r: r['stem'])
    ghosts, last_active, last_pool = [], {}, None
    for r in rs:
        scene = int(r['stem'].rsplit('_', 2)[1])
        pool = scene // 15
        if last_pool is not None and pool != last_pool:
            ghosts += list(last_active.values())
        last_pool = pool
        last_active = {o['name']: (o['x'], o['y']) for o in r['objects']}
        total += 1
        cx, cy, _, _, _, yaw = r['cam']
        for gx, gy in ghosts:
            d = math.hypot(gx - cx, gy - cy)
            ang = abs((math.atan2(gy - cy, gx - cx) - yaw + math.pi) % (2 * math.pi) - math.pi)
            if d < 30 and ang < HALF_FOV + math.atan2(2.5, max(d, 0.1)):
                drop.append(r)
                break
print(f'{ds}: {len(drop)} of {total} frames may show a ghost')
if apply:
    for r in drop:
        for kind, ext in (('images', 'jpg'), ('labels', 'txt')):
            src = f"{ds}/{kind}/{r['split']}/{r['stem']}.{ext}"
            dst = f"{ds}/quarantine/{kind}/{r['split']}"
            os.makedirs(dst, exist_ok=True)
            if os.path.exists(src):
                shutil.move(src, dst)
    print(f'moved to {ds}/quarantine/')
