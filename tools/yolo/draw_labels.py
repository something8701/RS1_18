#!/usr/bin/env python3
# draw_labels.py: DATASET OUT.jpg [N] [GLOB]: grid of N images with their YOLO boxes drawn
import glob
import sys

import cv2
import numpy as np

COL = [(0, 200, 255), (255, 120, 0), (0, 0, 255)]
NAMES = ['stump', 'log', 'bag']
ds, out = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
paths = sorted(glob.glob(f'{ds}/images/*/{sys.argv[4] if len(sys.argv) > 4 else "*"}.jpg'))[:n]
ims = []
for p in paths:
    im = cv2.imread(p)
    h, w = im.shape[:2]
    for line in open(p.replace('/images/', '/labels/').replace('.jpg', '.txt')):
        c, x, y, bw, bh = map(float, line.split()[:5])
        c = int(c)
        x0, y0 = int((x - bw / 2) * w), int((y - bh / 2) * h)
        x1, y1 = int((x + bw / 2) * w), int((y + bh / 2) * h)
        cv2.rectangle(im, (x0, y0), (x1, y1), COL[c], 1)
        cv2.putText(im, NAMES[c], (x0, max(y0 - 2, 8)), 0, 0.4, COL[c], 1)
    cv2.putText(im, p.split('/')[-1], (5, 15), 0, 0.45, (0, 0, 0), 1)
    ims.append(im)
while len(ims) % 2:
    ims.append(np.zeros_like(ims[0]))
cv2.imwrite(out, np.vstack([np.hstack(ims[i:i + 2]) for i in range(0, len(ims), 2)]))
