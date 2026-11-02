#!/usr/bin/env python3
# make_maths_figures.py: Draw the figures of mac/MATHS_GUIDE.html from the project's own code and data
import argparse
import base64
import csv
import html
import json
import math
import os
import re
import sys
from dataclasses import fields

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, '..', '..'))
WS = os.path.normpath(os.path.join(SRC, '..', '..'))
sys.path.insert(0, os.path.join(SRC, 'deforestation_monitoring'))

from deforestation_monitoring import radio_model, tree_detection
from deforestation_monitoring.waste import BAG_RADII

DATA = os.path.join(WS, 'eval', 'figures')
F_HUSKY = 360 / math.tan(2.0944 / 2)


def esc(s):
    return html.escape(str(s), quote=False)


class Svg:

    def __init__(self, w, h, label):
        self.w, self.h, self.label, self.e = w, h, label, []

    def line(self, x1, y1, x2, y2, cls='ink', **a):
        self.e.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                      f'class="{cls}"{_attrs(a)}/>')

    def rect(self, x, y, w, h, cls='fsoft', **a):
        self.e.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" '
                      f'height="{max(h, 0):.1f}" class="{cls}"{_attrs(a)}/>')

    def circle(self, cx, cy, r, cls='ffg', **a):
        self.e.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" class="{cls}"{_attrs(a)}/>')

    def poly(self, pts, cls='ink', closed=False, **a):
        p = ' '.join(f'{x:.1f},{y:.1f}' for x, y in pts)
        tag = 'polygon' if closed else 'polyline'
        self.e.append(f'<{tag} points="{p}" class="{cls}"{_attrs(a)}/>')

    def path(self, d, cls='ink', **a):
        self.e.append(f'<path d="{d}" class="{cls}"{_attrs(a)}/>')

    def text(self, x, y, s, cls='', anchor='start', raw=False, **a):
        c = f' class="{cls}"' if cls else ''
        an = '' if anchor == 'start' else f' text-anchor="{anchor}"'
        self.e.append(f'<text x="{x:.1f}" y="{y:.1f}"{c}{an}{_attrs(a)}>'
                      f'{s if raw else esc(s)}</text>')

    def arrow(self, x1, y1, x2, y2, cls='ink', head=7, **a):
        self.line(x1, y1, x2, y2, cls, **a)
        ang = math.atan2(y2 - y1, x2 - x1)
        pts = [(x2, y2)] + [(x2 - head * math.cos(ang + s * 0.42), y2 - head * math.sin(ang + s * 0.42))
                            for s in (1, -1)]
        fill = {'ink': 'ffg', 'accl': 'facc', 'dr': 'fdrs', 'hu': 'fhus', 'okl': 'fok',
                'badl': 'fbad', 'viol': 'fvios', 'ax': 'fmt'}.get(cls.split()[0], 'ffg')
        self.poly(pts, fill, closed=True)

    def raw(self, s):
        self.e.append(s)

    def svg(self):
        return (f'<svg viewBox="0 0 {self.w} {self.h}" role="img" aria-label="{esc(self.label)}">'
                + ''.join(self.e) + '</svg>')


def _attrs(a):
    return ''.join(f' {k.replace("_", "-")}="{v}"' for k, v in a.items())


class Ax:

    def __init__(self, s, x, y, w, h, xr, yr, xlog=False, ylog=False):
        self.s, self.x, self.y, self.w, self.h = s, x, y, w, h
        self.xr, self.yr, self.xlog, self.ylog = xr, yr, xlog, ylog

    def _n(self, v, r, log):
        if log:
            return (math.log10(v) - math.log10(r[0])) / (math.log10(r[1]) - math.log10(r[0]))
        return (v - r[0]) / (r[1] - r[0])

    def X(self, v):
        return self.x + self._n(v, self.xr, self.xlog) * self.w

    def Y(self, v):
        return self.y + self.h - self._n(v, self.yr, self.ylog) * self.h

    def frame(self, xt, yt, xl='', yl='', xf=None, yf=None, grid=True, yright=False):
        s = self.s
        xf = xf or _fmt
        yf = yf or _fmt
        for v in yt:
            if grid:
                s.line(self.x, self.Y(v), self.x + self.w, self.Y(v), 'grid-l')
            tx = self.x + self.w + 6 if yright else self.x - 6
            s.text(tx, self.Y(v) + 4, yf(v), 'tk', 'start' if yright else 'end')
        for v in xt:
            if grid:
                s.line(self.X(v), self.y, self.X(v), self.y + self.h, 'grid-l')
            s.text(self.X(v), self.y + self.h + 17, xf(v), 'tk', 'middle')
        s.line(self.x, self.y + self.h, self.x + self.w, self.y + self.h, 'ax')
        s.line(self.x, self.y, self.x, self.y + self.h, 'ax')
        if xl:
            s.text(self.x + self.w / 2, self.y + self.h + 32, xl, 'mt', 'middle')
        if yl:
            cx = self.x + self.w + 44 if yright else self.x - 40
            cy = self.y + self.h / 2
            s.text(cx, cy, yl, 'mt', 'middle', transform=f'rotate(-90 {cx:.1f} {cy:.1f})')

    def curve(self, xs, ys, cls, **a):
        pts = list(zip(xs, ys))
        step = max(1, len(pts) // 150)
        pts = pts[::step] + ([pts[-1]] if (len(pts) - 1) % step else [])
        self.s.poly([(self.X(x), self.Y(y)) for x, y in pts], cls, **a)


def _fmt(v):
    if isinstance(v, str):
        return v
    if abs(v) >= 1000:
        return f'{v:,.0f}'
    return f'{v:g}'


def fig(svg, caption):
    return f'<div class="svgbox">{svg.svg()}<p class="cap">{caption}</p></div>'


TREES_X = [(-8.0, 6.2, 3.2, 2.4, 'oak'), (2.6, 4.6, 1.6, 0.9, 'pine'), (9.5, 6.0, 3.0, 2.2, 'oak')]


def surface(x):
    z = 0.0
    for xc, h, r, base, kind in TREES_X:
        u = abs(x - xc) / r
        if u < 1:
            z = max(z, base + (h - base) * math.sqrt(1 - u * u) if kind == 'oak'
                    else h - (h - base) * u)
    return z


def raycast(theta, alt=10.0, step=0.01, rmax=40.0):
    t = 0.0
    while t < rmax:
        x, z = t * math.sin(theta), alt - t * math.cos(theta)
        if z <= surface(x):
            return x, max(z, 0.0)
        t += step
    return None


def fig_pushbroom():
    s = Svg(860, 330, 'Ray-cast push-broom scan over three trees and the canopy height it produces')
    a = Ax(s, 70, 30, 620, 240, (-16, 16), (0, 11))
    a.frame(range(-16, 17, 4), range(0, 11, 2), 'across-track distance from the drone (m)', 'height (m)')
    xs = np.linspace(-16, 16, 641)
    a.curve(xs, [surface(x) for x in xs], 'ax')
    for xc, h, r, base, kind in TREES_X:
        s.line(a.X(xc), a.Y(0), a.X(xc), a.Y(base), 'ink', stroke_width=2)
    hits = []
    for deg in np.arange(-60, 60.01, 1.0):
        hp = raycast(math.radians(deg))
        if hp:
            hits.append(hp)
            if round(deg) % 6 == 0:
                s.line(a.X(0), a.Y(10), a.X(hp[0]), a.Y(hp[1]), 'dr', opacity='.45')
    cells = {}
    for x, z in hits:
        c = math.floor(x / 0.5)
        cells[c] = max(cells.get(c, -1), z)
    run = []
    for c in list(range(-32, 32)) + [None]:
        if c is not None and c in cells:
            z = cells[c]
            run += [(a.X(c * 0.5), a.Y(z)), (a.X(c * 0.5 + 0.5), a.Y(z))]
            continue
        if len(run) > 1:
            s.poly(run, 'hu', stroke_width=3, opacity='.9')
        run = []
        if c is not None:
            s.rect(a.X(c * 0.5), a.Y(0) - 4, a.X(c * 0.5 + 0.5) - a.X(c * 0.5), 4, 'fbad', opacity='.7')
    for x, z in hits:
        s.circle(a.X(x), a.Y(z), 2.2, 'fdrs')
    s.rect(a.X(0) - 12, a.Y(10) - 6, 24, 10, 'fsoft', rx='3')
    s.text(a.X(0) + 18, a.Y(10) + 2, 'drone, 10 m up')
    lx = 712
    s.line(lx, 60, lx + 22, 60, 'dr', opacity='.6'); s.text(lx + 28, 64, 'beams (1° apart,')
    s.text(lx + 28, 78, 'every 6th drawn)')
    s.circle(lx + 11, 100, 3, 'fdrs'); s.text(lx + 28, 104, 'hit: first surface')
    s.line(lx, 126, lx + 22, 126, 'hu', stroke_width=3); s.text(lx + 28, 130, 'CHM: highest hit')
    s.text(lx + 28, 144, 'per cell')
    s.rect(lx, 162, 22, 5, 'fbad', opacity='.7'); s.text(lx + 28, 168, 'no hit: unknown (−1)')
    s.line(lx, 190, lx + 22, 190, 'ax'); s.text(lx + 28, 194, 'true canopy')
    n_hit = len(hits)
    return fig(s, f'Computed: {n_hit} beams 1° apart are ray-cast from 10 m over an oak, a pine and an oak. '
                  'Each beam stops at the first surface it meets. The highest hit per cell is the canopy '
                  'height model. Red marks: cells no beam reached because a crown hides them from this '
                  'pass; they stay "unknown" until another strip sees them, which is why the drone flies '
                  'overlapping strips.')


def fig_pinhole():
    s = Svg(860, 280, 'Pinhole camera: the same object twice as far looks half as tall')
    cx, cy, f = 70, 125, 80
    d1, S = 300, 130
    s.line(cx, cy, 820, cy, 'ax', stroke_dasharray='5 4')
    s.line(cx + f, cy - 70, cx + f, cy + 70, 'ink')
    for d, cls, lab in ((d1, 'accl', 'd'), (2 * d1, 'dr', '2d')):
        x = cx + d
        s.line(x, cy - S / 2, x, cy + S / 2, cls.replace('l', 'l'), stroke_width=4)
        s.line(cx, cy, x, cy - S / 2, cls, opacity='.6')
        s.line(cx, cy, x, cy + S / 2, cls, opacity='.6')
        hh = f * (S / 2) / d
        s.line(cx + f + (4 if lab == 'd' else -4), cy - hh, cx + f + (4 if lab == 'd' else -4), cy + hh,
               cls, stroke_width=5)
        s.text(x, cy + S / 2 + 18, f'object S at {lab}', 'mt', 'middle')
    s.circle(cx, cy, 5, 'ffg')
    s.text(cx - 6, cy + 22, 'camera', 'mt', 'middle')
    s.text(cx + f, cy - 78, 'image plane', 'mt', 'middle')
    s.text(cx + f / 2, cy - 6, 'f', 'mt', 'middle')
    s.text(40, 230, 'similar triangles:  s / f = S / d', 'b')
    s.text(40, 250, 'size on the image  s = f · S / d', '')
    s.text(40, 270, 'twice as far → half as tall, a quarter of the pixels')
    s.text(cx + f + 12, cy - 40, 's at d', 'accl-t')
    return fig(s, 'The pinhole model with the Husky camera\'s geometry in mind (f = 207.8 px). The blue object '
                  'at distance d and the orange one at 2d are the same size; the orange image is half as '
                  'tall. Area on the image falls with 1/d², which the bag-size check uses.')


def fig_change_timeline():
    s = Svg(860, 312, 'One map cell: readings per scan, the ground streak and when LOST fires')
    seq = 'CCCGCCCC' + 'CCCCGC..' + 'G.GGGGG'
    freeze = 8
    n = len(seq)
    x0, dx = 90, 30
    k = sum(1 for c in seq[:freeze] if c == 'C')
    nb = sum(1 for c in seq[:freeze] if c != '.')
    frac = k / nb
    streak, streaks, lost_at = 0, [], None
    for i, c in enumerate(seq):
        if c == 'G':
            streak += 1
        elif c == 'C':
            streak = 0
        streaks.append(streak)
        if i >= freeze and lost_at is None and frac >= 0.6 and c == 'G' and streak >= 3:
            lost_at = i
    s.text(14, 64, 'reading', 'mt')
    for i, c in enumerate(seq):
        x = x0 + i * dx
        if c == '.':
            s.rect(x + 3, 46, dx - 6, 26, 'fcard', stroke_dasharray='3 3')
        else:
            s.rect(x + 3, 46, dx - 6, 26, 'fok' if c == 'C' else 'fwarn', opacity='.8' if c == 'C' else '.55')
        s.text(x + dx / 2, 90, str(i + 1), 'tk', 'middle')
    s.text(x0 - 6, 104, 'scan', 'tk', 'end')
    a = Ax(s, x0, 130, n * dx, 110, (0, n), (0, 4))
    a.frame([], [0, 1, 2, 3], '', '')
    s.text(14, 190, 'ground', 'mt'); s.text(14, 204, 'streak', 'mt')
    pts = []
    for i, st in enumerate(streaks):
        pts += [(a.X(i), a.Y(st)), (a.X(i + 1), a.Y(st))]
    s.poly(pts, 'hu', stroke_width=2.5)
    s.line(a.X(0), a.Y(3), a.X(n), a.Y(3), 'badl', stroke_dasharray='5 4')
    s.text(a.X(n) + 6, a.Y(3) + 4, 'need 3', 'bad')
    fx = x0 + freeze * dx
    s.line(fx, 30, fx, 250, 'viol', stroke_width=2)
    s.text(fx - 6, 24, f'baseline frozen: canopy in {k} of {nb} scans = {frac:.2f} ≥ 0.6', 'vio', 'end')
    s.text(x0 + 3.5 * dx, 284, 'glitch before the freeze:', 'mt', 'middle')
    s.text(x0 + 3.5 * dx, 298, 'only lowers the fraction', 'mt', 'middle')
    s.text(x0 + 12.5 * dx, 284, 'one ground read after:', 'mt', 'middle')
    s.text(x0 + 12.5 * dx, 298, 'streak 1, then reset', 'mt', 'middle')
    s.text(x0 + 15 * dx, 120, 'not seen: streak kept', 'mt', 'middle')
    cut = x0 + 16 * dx
    s.line(cut, 40, cut, 78, 'badl', stroke_width=2)
    s.text(cut + 4, 36, 'tree cut', 'bad')
    lx = x0 + (lost_at + 0.5) * dx
    s.arrow(lx, 262, lx, 246, 'badl')
    s.text(lx, 284, f'LOST (−100) at scan {lost_at + 1}', 'bad b', 'middle')
    return fig(s, 'The real rule, run on one cell: green = canopy (above 2 m), orange = ground, dashed = the '
                  'drone was elsewhere. Unobserved scans neither add to the streak nor reset it, and a single '
                  'ground reading resets at the next canopy reading. Only 3 ground readings after the cut, on a '
                  'cell that was canopy in ≥ 60 % of its baseline scans, make it LOST.')


def _live_params():
    import yaml
    with open(os.path.join(SRC, 'deforestation_monitoring', 'config', 'tree_detection_params.yaml')) as fh:
        y = yaml.safe_load(fh)['/**']['ros__parameters']
    names = {f.name for f in fields(tree_detection.DetectionParams)}
    return tree_detection.DetectionParams(**{k: v for k, v in y.items() if k in names})


def _dense():
    m = np.load(os.path.join(DATA, 'dense_chm.npy'))
    return m, 0.25, -40.0, -40.0


def fig_otsu():
    m, *_ = _dense()
    p = _live_params()
    obs = m[m >= 0]
    t_otsu = tree_detection.otsu_threshold(obs, include_zero=True)
    g = float(np.percentile(obs, p.ground_percentile * 100))
    if g > p.ground_floor + p.ground_margin:
        g = p.ground_floor
    T = max(p.min_height, g + p.ground_margin, min(t_otsu, p.auto_threshold_cap))
    hist, edges = np.histogram(obs, bins=256)
    cen = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist); w1 = hist.sum() - w0
    cm = np.cumsum(hist * cen)
    ok = (w0 > 0) & (w1 > 0)
    sb = np.zeros_like(cen)
    sb[ok] = w0[ok] * w1[ok] * (cm[ok] / w0[ok] - (cm[-1] - cm[ok]) / w1[ok]) ** 2
    sb /= sb.max()
    s = Svg(860, 320, 'Height histogram of a recorded live map with the Otsu curve and the final threshold')
    a = Ax(s, 70, 26, 600, 230, (0, 7.5), (1, 1e5), ylog=True)
    a.frame(range(0, 8), [1, 10, 100, 1000, 10000, 100000], 'height (m)', 'cells (log scale)')
    hb, he = np.histogram(obs, bins=np.arange(0, 7.6, 0.1))
    for c, e0 in zip(hb, he[:-1]):
        if c > 0:
            s.rect(a.X(e0) + 0.5, a.Y(c), a.X(e0 + 0.1) - a.X(e0) - 1, a.Y(1) - a.Y(c), 'fhus', opacity='.75')
    a2 = Ax(s, 70, 26, 600, 230, (0, 7.5), (0, 1.05))
    a2.curve(cen, sb, 'viol', stroke_width=2.5)
    s.text(a2.X(cen[int(np.argmax(sb))]) + 6, a2.Y(1.0) + 4, 'between-class variance σB²(t)', 'vio')
    for v, cls, lab, dy in ((t_otsu, 'viol', f'Otsu t* = {t_otsu:.2f} m', 0),
                            (T, 'badl', f'used: T = {T:.2f} m', 16)):
        s.line(a.X(v), a.y, a.X(v), a.y + a.h, cls, stroke_dasharray='6 4', stroke_width=2)
    lx = 690
    rows = [('ground level g (20th percentile)', f'{g:.2f} m'), ('g + margin 0.5 m', f'{g + 0.5:.2f} m'),
            ('Otsu t*', f'{t_otsu:.2f} m'), ('Otsu, capped at 1.5 m', f'{min(t_otsu, 1.5):.2f} m'),
            ('floor (min_height)', '1.50 m')]
    s.text(lx, 40, 'T = max of:', 'b')
    for i, (k_, v_) in enumerate(rows):
        s.text(lx, 62 + i * 30, k_, 'mt')
        s.text(lx, 76 + i * 30, v_)
    s.text(lx, 222, f'→ T = {T:.2f} m', 'bad b')
    gfrac = float(np.mean(obs < T))
    return fig(s, f'Real data: the {obs.size:,} scanned cells of a recorded live dense-forest map '
                  f'(bags/dense_survey). The tall bar at 0 m is ground ({gfrac:.0%} of cells are below T), the '
                  'broad hump is canopy. Purple: Otsu\'s between-class variance; its peak is the best split. '
                  'Here t* lands inside the canopy hump, exactly the drift the 1.5 m cap guards against; the '
                  'threshold used is the red line.')


def fig_saddle():
    s = Svg(860, 290, 'Saddle test on two crown profiles: a shallow dip merges, a deep valley splits')

    def prof(x, a, b):
        return max(a[1] * math.exp(-((x - a[0]) / a[2]) ** 2), b[1] * math.exp(-((x - b[0]) / b[2]) ** 2))

    cases = [((-1.0, 6.2, 2.6), (1.2, 5.8, 2.4), 'one oak with two lobes'),
             ((-2.2, 6.2, 1.9), (2.4, 4.6, 1.3), 'an oak and a pine')]
    for i, (A, B, title) in enumerate(cases):
        a = Ax(s, 60 + i * 420, 36, 340, 190, (-5, 5), (0, 7))
        a.frame(range(-4, 5, 2), range(0, 8, 2), 'distance along the line (m)', 'height (m)' if i == 0 else '')
        xs = np.linspace(-5, 5, 401)
        zs = [prof(x, A, B) for x in xs]
        between = [(x, z) for x, z in zip(xs, zs) if A[0] <= x <= B[0]]
        xsad, zsad = min(between, key=lambda p: p[1])
        thr = 0.75 * min(A[1], B[1])
        merge = zsad >= thr
        if not merge:
            left = [(a.X(x), a.Y(z)) for x, z in zip(xs, zs) if x <= xsad]
            right = [(a.X(x), a.Y(z)) for x, z in zip(xs, zs) if x >= xsad]
            s.poly(left + [(a.X(xsad), a.Y(0)), (a.X(-5), a.Y(0))], 'fdrs', closed=True, opacity='.3')
            s.poly(right + [(a.X(5), a.Y(0)), (a.X(xsad), a.Y(0))], 'fhus', closed=True, opacity='.3')
        a.curve(xs, zs, 'okl' if merge else 'ink', stroke_width=2.5)
        s.line(a.X(-5), a.Y(thr), a.X(5), a.Y(thr), 'badl', stroke_dasharray='6 4')
        s.circle(a.X(xsad), a.Y(zsad), 5, 'fok' if merge else 'fbad')
        s.text(a.X(xsad), a.Y(zsad) + (-10 if merge else 20), f'saddle {zsad:.1f} m', 'b', 'middle')
        for P in (A, B):
            s.text(a.X(P[0]), a.Y(P[1]) - 8, f'{P[1]} m', 'tk', 'middle')
        s.text(a.x + 2, 34, f'red line: 0.75 × {min(A[1], B[1])} = {thr:.2f} m', 'bad tk')
        s.text(a.x, 20, title, 'b')
        verdict = 'saddle ≥ line → ONE tree (merge)' if merge else 'saddle < line → TWO trees'
        s.text(a.x + a.w / 2, 280, verdict, 'ok b' if merge else 'bad b', 'middle')
    return fig(s, 'The saddle test with real numbers: the lowest point on the straight line between two '
                  'treetops is compared with 0.75 × the lower top. Left: the dip stays high, so the two bumps '
                  'are one crown. Right: a deep valley, so two trees, and the watershed gives each the cells on '
                  'its side (orange, blue).')


def _colour_labels(labels):
    pal = ['fdrs', 'fhus', 'fvios', 'fok', 'fwarn', 'facc']
    adj = {}
    for a_, b_ in ((labels[1:, :], labels[:-1, :]), (labels[:, 1:], labels[:, :-1])):
        m = (a_ != b_) & (a_ > 0) & (b_ > 0)
        for u, v in zip(a_[m], b_[m]):
            adj.setdefault(int(u), set()).add(int(v)); adj.setdefault(int(v), set()).add(int(u))
    col = {}
    for lab in sorted(set(np.unique(labels)) - {0}):
        used = {col.get(n) for n in adj.get(int(lab), ())}
        col[int(lab)] = next(c for c in pal if c not in used) if len(used) < len(pal) else pal[0]
    return col


_PAL_RGB = {'fdrs': (232, 113, 10), 'fhus': (10, 158, 199), 'fvios': (137, 68, 171), 'fok': (36, 138, 61),
            'fwarn': (178, 80, 0), 'facc': (0, 113, 227)}


def _rgba_uri(img):
    import cv2
    ok, buf = cv2.imencode('.png', img[..., [2, 1, 0, 3]])
    return 'data:image/png;base64,' + base64.b64encode(buf.tobytes()).decode()


def _itd_state():
    from scipy import ndimage
    m, res, ox, oy = _dense()
    p = _live_params()
    scanned = m >= 0
    chm = np.where(scanned, m, 0.0).astype(np.float32)
    chm_s = tree_detection.smooth_chm(chm, scanned, p.smooth_sigma / res)
    chm_s = np.where(scanned, ndimage.grey_closing(chm_s, size=p.pit_fill_size), 0.0).astype(np.float32)
    mask0, T = tree_detection.canopy_mask_from_chm(chm_s, scanned, p)
    mask = tree_detection.refine_canopy_mask(mask0, p)
    narrow = tree_detection.smooth_chm(chm, scanned, p.narrow_sigma / res)
    heights = np.where(mask, chm_s, 0.0)
    rad = np.clip(np.ceil(np.clip(p.window_scale * heights + p.window_offset, p.min_window, p.max_window)
                          / res), 1, 32).astype(int)
    cand = np.zeros(chm.shape, bool)
    for r in np.unique(rad):
        cand |= (rad == r) & (heights == ndimage.maximum_filter(heights, size=2 * r + 1, mode='constant'))
    cand &= mask
    markers = tree_detection.treetop_markers(chm_s, mask, p, raw_chm=chm, narrow_chm=narrow)
    dets, labels, _ = tree_detection.detect_trees(chm, scanned, p, ox, oy, return_debug=True)
    truth = tree_detection.parse_tree_truth(os.path.join(SRC, '41068_ignition_bringup', 'worlds',
                                                         'dense_forest.sdf'))
    W_, H_ = 18.0, 12.0
    best = None
    for cx in np.arange(-24, 24 - W_ + 0.1, 1.0):
        for cy in np.arange(-24, 24 - H_ + 0.1, 1.0):
            k = sum(1 for d in dets if cx <= d.x < cx + W_ and cy <= d.y < cy + H_)
            if best is None or k > best[0]:
                best = (k, cx, cy)
    _, wx0, wy0 = best
    win = (int((wx0 - ox) / res), int((wy0 - oy) / res), int(W_ / res), int(H_ / res))
    return dict(m=m, res=res, ox=ox, oy=oy, p=p, scanned=scanned, chm=chm, chm_s=chm_s, mask0=mask0,
                mask=mask, T=T, cand=cand, markers=markers, dets=dets, labels=labels, truth=truth,
                win=win, wx0=wx0, wy0=wy0, W=W_, H=H_)


_ST = {}


def _st():
    if not _ST:
        _ST.update(_itd_state())
    return _ST


def _crop_img(st, fn):
    ix0, iy0, nx, ny = st['win']
    img = np.zeros((ny, nx, 4), np.uint8)
    for ix in range(ix0, ix0 + nx):
        for iy in range(iy0, iy0 + ny):
            img[iy0 + ny - 1 - iy, ix - ix0] = fn(ix, iy)
    return img


def _height_rgba(h, scanned_cell, top=6.5):
    if not scanned_cell:
        return (215, 60, 60, 40)
    return (36, 138, 61, int(15 + 230 * min(max(h, 0), top) / top))


def _label_rgba(labels, col_map):
    def f(ix, iy):
        lab = int(labels[ix, iy])
        if lab <= 0:
            return (0, 0, 0, 0)
        return (*_PAL_RGB.get(col_map.get(lab, 'fhus'), (120, 120, 120)), 140)
    return f


def _panel(s, st, x, y, w, img, title):
    ix0, iy0, nx, ny = st['win']
    h = w * ny / nx
    if title:
        s.text(x, y - 6, title, 'b tk')
    s.raw(f'<image href="{_rgba_uri(img)}" x="{x}" y="{y}" width="{w}" height="{h:.1f}" '
          'style="image-rendering:pixelated"/>')
    s.rect(x, y, w, h, 'ax', fill='none')
    cell = w / nx

    def to_px(wx, wy):
        return x + (wx - st['wx0']) / st['res'] * cell, y + (st['wy0'] + st['H'] - wy) / st['res'] * cell

    def cell_px(ix, iy):
        return x + (ix - ix0 + 0.5) * cell, y + (iy0 + ny - 1 - iy + 0.5) * cell

    def inside(wx, wy):
        return st['wx0'] <= wx < st['wx0'] + st['W'] and st['wy0'] <= wy < st['wy0'] + st['H']

    def in_cell(ix, iy):
        return ix0 <= ix < ix0 + nx and iy0 <= iy < iy0 + ny
    return h, to_px, cell_px, inside, in_cell


def _boundaries(s, st, x, y, w, labels):
    ix0, iy0, nx, ny = st['win']
    cell = w / nx
    segs = []
    for ix in range(ix0, ix0 + nx):
        for iy in range(iy0, iy0 + ny):
            l0 = labels[ix, iy]
            if l0 <= 0:
                continue
            if ix + 1 < ix0 + nx and labels[ix + 1, iy] > 0 and labels[ix + 1, iy] != l0:
                segs.append(f'M{x + (ix + 1 - ix0) * cell:.1f},{y + (iy0 + ny - 1 - iy) * cell:.1f}v{cell:.1f}')
            if iy + 1 < iy0 + ny and labels[ix, iy + 1] > 0 and labels[ix, iy + 1] != l0:
                segs.append(f'M{x + (ix - ix0) * cell:.1f},{y + (iy0 + ny - 1 - iy) * cell:.1f}h{cell:.1f}')
    s.path(''.join(segs), 'ink', stroke_width=1.2)


def _tri(s, px, py, cls='ffg', r=4.5):
    s.poly([(px, py - r), (px - r, py + r * 0.8), (px + r, py + r * 0.8)], cls, closed=True)


def _x(s, px, py, cls='badl', r=3.5):
    s.path(f'M{px - r:.1f},{py - r:.1f}l{2 * r},{2 * r}M{px - r:.1f},{py + r:.1f}l{2 * r},{-2 * r}', cls,
           stroke_width=1.8)


def fig_itd_steps():
    st = _st()
    ix0, iy0, nx, ny = st['win']
    W, gx, gy = 268, 288, 236
    s = Svg(860, 500, 'The six steps of individual tree detection on a recorded live canopy map')
    pos = [(10 + c * gx, 30 + r * gy) for r in range(2) for c in range(3)]
    chm, chm_s, sc = st['chm'], st['chm_s'], st['scanned']
    x, y = pos[0]
    _panel(s, st, x, y, W, _crop_img(st, lambda i, j: _height_rgba(chm[i, j], sc[i, j])),
           '1 · raw height map (greener = taller)')
    x, y = pos[1]
    _panel(s, st, x, y, W, _crop_img(st, lambda i, j: _height_rgba(chm_s[i, j], sc[i, j])),
           '2 · smoothed, σ 0.9 m, pits filled')

    def mask_px(i, j):
        a0, a1 = st['mask0'][i, j], st['mask'][i, j]
        if a1 and a0:
            return (36, 138, 61, 150)
        if a0 and not a1:
            return (215, 0, 21, 230)
        if a1 and not a0:
            return (0, 113, 227, 230)
        return (0, 0, 0, 0)
    x, y = pos[2]
    _panel(s, st, x, y, W, _crop_img(st, mask_px), f'3 · canopy: height ≥ {st["T"]:.1f} m, cleaned')
    x, y = pos[3]
    h, to_px, cell_px, inside, in_cell = _panel(
        s, st, x, y, W, _crop_img(st, lambda i, j: _height_rgba(chm_s[i, j], sc[i, j])), '')
    nc = 0
    for i, j in np.argwhere(st['cand']):
        if in_cell(i, j):
            nc += 1
            s.circle(*cell_px(i, j), 2, 'fmt')
    nm = nn = 0
    for i, j in st['markers']:
        if in_cell(i, j):
            nm += 1
            nn += not st['cand'][i, j]
            _tri(s, *cell_px(i, j))
    s.text(x, y - 6, f'4 · {nm} treetops: {nm - nn} window maxima + {nn} pine peak{"s" * (nn != 1)}', 'b tk')
    x, y = pos[4]
    col = _colour_labels(st['labels'][ix0:ix0 + nx, iy0:iy0 + ny])
    _panel(s, st, x, y, W, _crop_img(st, _label_rgba(st['labels'], col)), '5 · watershed crowns')
    _boundaries(s, st, x, y, W, st['labels'])
    x, y = pos[5]
    h, to_px, cell_px, inside, in_cell = _panel(s, st, x, y, W, _crop_img(st, _label_rgba(st['labels'], col)), '')
    _boundaries(s, st, x, y, W, st['labels'])
    dets = [d for d in st['dets'] if inside(d.x, d.y)]
    tru = [t for t in st['truth'] if inside(t[1], t[2])]
    sc_ = tree_detection.score_detections(dets, tru, 1.5)
    for d in dets:
        s.circle(*to_px(d.x, d.y), 3.5, 'ffg')
    for t in tru:
        _x(s, *to_px(t[1], t[2]))
    s.text(x, y - 6, f'6 · {sc_["true_positives"]} of {len(tru)} trunks found, {sc_["false_positives"]} false (1.5 m)',
           'b tk')
    ly1, ly2 = 30 + 196, 30 + gy + 196
    s.rect(pos[2][0], ly1 - 9, 10, 10, 'fbad'); s.text(pos[2][0] + 14, ly1, 'removed (specks)', 'tk')
    s.rect(pos[2][0] + 120, ly1 - 9, 10, 10, 'facc'); s.text(pos[2][0] + 134, ly1, 'filled (gaps)', 'tk')
    s.circle(pos[3][0] + 5, ly2 - 4, 2, 'fmt'); s.text(pos[3][0] + 12, ly2, 'window maximum', 'tk')
    _tri(s, pos[3][0] + 124, ly2 - 4); s.text(pos[3][0] + 133, ly2, 'kept: NMS + saddle test', 'tk')
    s.circle(pos[5][0] + 5, ly2 - 4, 3.5, 'ffg'); s.text(pos[5][0] + 12, ly2, 'tree position', 'tk')
    _x(s, pos[5][0] + 100, ly2 - 4); s.text(pos[5][0] + 110, ly2, 'true trunk (world file)', 'tk')
    return fig(s, 'Real data, every step: the live detector\'s own functions run on a recorded dense-forest map '
                  '(18 × 12 m shown). 1 → 2: smoothing turns branch bumps into one hill per crown. 3: the height '
                  'threshold and the opening/closing clean the canopy mask. 4: each grey dot is the top of its own '
                  'window; non-maximum suppression and the saddle test keep one per tree. 5: crowns flood from '
                  'the kept tops. 6: positions against the world file. Most misses are pines under oak crowns, '
                  'which no top-down sensor sees.')


def fig_dbscan_vs_itd():
    from sklearn.cluster import DBSCAN
    st = _st()
    ix0, iy0, nx, ny = st['win']
    res, ox, oy = st['res'], st['ox'], st['oy']
    cells = np.argwhere(st['scanned'] & (st['chm'] >= 1.5))
    pts = np.c_[ox + (cells[:, 0] + .5) * res, oy + (cells[:, 1] + .5) * res]
    xs, ys = np.nonzero(st['scanned'])
    box = (ox + xs.min() * res + 3, ox + xs.max() * res - 3, oy + ys.min() * res + 3, oy + ys.max() * res - 3)

    def inb(x, y):
        return box[0] <= x <= box[1] and box[2] <= y <= box[3]
    truth = [t for t in st['truth'] if inb(t[1], t[2])]

    class D:
        pass

    def run(eps, ms):
        lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(pts)
        grid = np.zeros(st['chm'].shape, np.int32)
        grid[cells[:, 0], cells[:, 1]] = np.where(lab >= 0, lab + 1, -1)
        ds = []
        for l in set(lab) - {-1}:
            d = D()
            d.x, d.y = pts[lab == l].mean(0)
            if inb(d.x, d.y):
                ds.append(d)
        return grid, tree_detection.score_detections(ds, truth, 1.5), len(set(lab) - {-1})
    sweep = {}
    for eps in (0.26, 0.36, 0.5, 0.75, 1.0):
        for ms in (3, 5, 9, 15):
            sweep[(eps, ms)] = run(eps, ms)[1]['f1']
    best_f1 = max(sweep.values())
    itd = [d for d in st['dets'] if inb(d.x, d.y)]
    sc_itd = tree_detection.score_detections(itd, truth, 1.5)
    panels = []
    for eps, ms, name in ((0.26, 5, 'DBSCAN ε 0.26 m, minPts 5 (its best F1)'),
                          (1.0, 9, 'DBSCAN ε 1.0 m, minPts 9')):
        g, sco, n = run(eps, ms)
        panels.append((g, sco, n, name))
    s = Svg(860, 352, 'DBSCAN against the watershed detector on the same recorded map, scored against the world file')
    W = 268
    for k, (g, sco, n, name) in enumerate(panels + [(st['labels'], sc_itd, len(st['dets']), 'our detector (watershed)')]):
        x, y = 10 + k * 288, 34
        sub = g[ix0:ix0 + nx, iy0:iy0 + ny]
        col = _colour_labels(np.where(sub > 0, sub, 0))

        def f(i, j, g=g, col=col):
            v = int(g[i, j])
            if v == -1:
                return (110, 110, 115, 120)
            if v <= 0:
                return (0, 0, 0, 0)
            return (*_PAL_RGB.get(col.get(v, 'fhus'), (120, 120, 120)), 150)
        h, to_px, cell_px, inside, in_cell = _panel(s, st, x, y, W, _crop_img(st, f), name)
        if k == 2:
            _boundaries(s, st, x, y, W, st['labels'])
        for t in st['truth']:
            if inside(t[1], t[2]):
                _x(s, *to_px(t[1], t[2]))
        s.text(x, y + h + 22, f'F1 {sco["f1"]:.2f}', ('ok' if k == 2 else 'bad') + ' b')
        s.text(x + 62, y + h + 22, f'P {sco["precision"]:.2f} · R {sco["recall"]:.2f}', 'tk')
        s.text(x, y + h + 40, f'{n} clusters on the whole map, {sco["false_positives"]} false', 'tk')
    s.text(10, 304, f'Scored on the whole map: {len(truth)} true trees, one-to-one within 1.5 m. DBSCAN tried at '
                    f'{len(sweep)} settings (ε 0.26–1.0 m, minPts 3–15): best F1 {best_f1:.2f}.', 'mt')
    s.text(10, 322, 'Same colour = one cluster. At any ε above one cell (0.25 m) touching crowns chain into one blob;', 'mt')
    s.text(10, 340, 'its other clusters are scraps where the canopy is broken. Red × = true trunks.', 'mt')
    return fig(s, 'Measured on the same live map. Left and middle: DBSCAN turns the continuous canopy into one '
                  'cluster spanning many trees, at its best setting and at a large one alike, because the cells are '
                  'just as dense across the join between two crowns as inside each. Right: the watershed uses height '
                  'and cuts at the valleys between crowns, more than doubling F1.')


def fig_f1():
    s = Svg(860, 312, 'F1 contours on the precision-recall plane with example operating points')
    a = Ax(s, 70, 20, 250, 250, (0, 1), (0, 1))
    a.frame([0, 0.25, 0.5, 0.75, 1], [0, 0.25, 0.5, 0.75, 1], 'precision P', 'recall R')
    for F in (0.2, 0.4, 0.6, 0.8, 0.9):
        ps = np.linspace(F / 2 + 1e-3, 1, 200)
        rs = F * ps / (2 * ps - F)
        keep = rs <= 1
        a.curve(ps[keep], rs[keep], 'viol', opacity='.7')
        s.text(a.X(1) + 4, a.Y(F / (2 - F)) + 4, f'F1 {F}', 'tk vio-t')
    pts = [(1.0, 0.1, 'everything certain, little found', 'bad'),
           (1.0, 0.98, 'synthetic dense (P 1.00, R 0.98)', 'ok'),
           (0.74, 0.53, 'early live dense (P 0.74, R 0.53)', 'warn')]
    for P_, R_, lab, c in pts:
        s.circle(a.X(P_), a.Y(R_), 5, {'bad': 'fbad', 'ok': 'fok', 'warn': 'fwarn'}[c])
    lx = 400
    s.text(lx, 40, 'F1 = 2PR / (P + R): the harmonic mean', 'b')
    s.text(lx, 70, 'Red point: P = 1.00, R = 0.10', 'bad')
    s.text(lx, 88, 'arithmetic mean (P + R)/2 = 0.55: looks fine', 'mt')
    s.text(lx, 106, 'F1 = 2·1·0.1 / 1.1 = 0.18: exposed', 'mt')
    s.text(lx, 136, 'Green: synthetic calibration, F1 0.99', 'ok')
    s.text(lx, 166, 'Orange: an early live dense-forest tune, F1 0.62', 'warn')
    s.text(lx, 196, 'The contours bend towards the low side: to score', 'mt')
    s.text(lx, 214, 'well you must find trees AND not invent them.', 'mt')
    return fig(s, 'Each purple curve joins the (precision, recall) pairs with the same F1. Because F1 is a '
                  'harmonic mean, it stays low unless both are high, so the calibration cannot win by '
                  'reporting only obvious trees (high P, low R) or by flooding the map with fakes (high R, low P).')


def fig_gates():
    s = Svg(860, 270, 'Fixed association gates overlap between close trees; adaptive gates never do')
    sc = 55
    trees = [(0.0, 0.0, 'A'), (1.4, 0.0, 'B'), (4.6, 0.2, 'C')]
    for panel, title in ((0, 'fixed 1.5 m gates'), (1, 'adaptive: r = min(0.4 · nearest, 1.5 m)')):
        ox_, oy_ = 100 + panel * 420, 130
        s.text(20 + panel * 420, 22, title, 'b')
        pos = [(ox_ + x * sc, oy_ - y * sc) for x, y, _ in trees]
        for i, (x, y, name) in enumerate(trees):
            nn = min(math.hypot(x - u, y - v) for u, v, _ in trees if (u, v) != (x, y))
            r = 1.5 if panel == 0 else min(0.4 * nn, 1.5)
            X, Y = pos[i]
            s.circle(X, Y, r * sc, 'dr' if panel else 'badl', stroke_dasharray='5 4', fill='none')
            s.circle(X, Y, 5, 'ffg')
            s.text(X, Y - 10, name, 'b', 'middle')
            s.text(X, Y + r * sc + 16, f'r = {r:.2f} m', 'tk', 'middle')
        if panel == 0:
            s.text(ox_ + 0.7 * sc, oy_ + 4, 'overlap', 'bad b', 'middle')
            s.text(20, oy_ + 124, 'a detection here fits A and B: an ID swap can happen', 'bad')
        else:
            s.text(440, oy_ + 124, 'r_A + r_B ≤ 0.8 · d < d: the gates can never touch', 'ok')
        s.line(pos[0][0], oy_ - 40, pos[1][0], oy_ - 40, 'ax')
        s.text((pos[0][0] + pos[1][0]) / 2, oy_ - 46, 'd = 1.4 m', 'tk', 'middle')
    return fig(s, 'To scale. Left: fixed 1.5 m gates make the area between two close trees ambiguous. Right: '
                  'the tracker\'s adaptive gates shrink in tight groups (0.56 m for A and B, 1.4 m apart) and grow '
                  'for a tree with more room (C), capped at 1.5 m, so every detection can belong to at most one '
                  'tree. The Hungarian algorithm then picks the optimal one-to-one assignment.')


def fig_lost_logic():
    s = Svg(860, 330, 'Logic of the LOST decision: evidence, gates, persistence and the merge hold')

    def box(x, y, w, h, lines, cls='fcard', tcls=''):
        s.rect(x, y, w, h, cls, rx='10')
        for i, t in enumerate(lines):
            s.text(x + 10, y + 18 + i * 15, t, tcls if i == 0 else 'mt')

    def gate(x, y, lab):
        s.rect(x, y, 46, 30, 'fsoft', rx='15', stroke='var(--line2)')
        s.text(x + 23, y + 19, lab, 'b', 'middle')
    box(10, 20, 220, 50, ['lost or dropped cells within r', '≥ 6  (r = crown, 0.6–2.0 m)'], tcls='b')
    box(10, 80, 220, 64, ['top fell', 'Δmax ≥ 0.3 m and 90 % of top', 'cells down ≥ 0.6 m, or Δmean ≥ 2 m'], tcls='b')
    box(10, 166, 220, 50, ['camera saw through it', 'two visits, seen-canopy < 0.3'], tcls='b')
    box(10, 226, 220, 50, ['height map agrees', 'the spot dropped'], tcls='b')
    gate(260, 65, 'AND'); gate(260, 205, 'AND')
    for y in (45, 112):
        s.arrow(230, y, 258, 80, 'ax')
    for y in (191, 251):
        s.arrow(230, y, 258, 220, 'ax')
    s.text(262, 58, 'height', 'tk'); s.text(262, 198, 'camera', 'tk')
    gate(340, 135, 'OR')
    s.arrow(306, 80, 338, 145, 'ax'); s.arrow(306, 220, 338, 155, 'ax')
    box(330, 20, 170, 50, ['detector lost it', 'unmatched this tick'], tcls='b')
    gate(430, 135, 'AND')
    s.arrow(386, 150, 428, 150, 'ax'); s.arrow(453, 70, 453, 133, 'ax')
    box(510, 120, 150, 60, ['streak + 1', 'any tick without it:', 'streak reset to 0'], 'fsoft', 'b')
    s.arrow(476, 150, 508, 150, 'ax')
    box(690, 120, 160, 60, ['streak ≥ 3 ticks?', 'no: wait for the', 'next tick'], 'fsoft', 'b')
    s.arrow(660, 150, 688, 150, 'ax')
    box(510, 220, 340, 64, ['oak? hold 30 s (240 s if a bigger oak stands near)',
                            'a bigger felled oak within 5.5 m in that window:',
                            'MERGED into it, not a second cut. Pines never merge.'], 'fsoft', 'b')
    s.arrow(770, 180, 770, 218, 'ax')
    s.rect(560, 296, 240, 30, 'fbad', rx='15', opacity='.85')
    s.text(680, 316, 'LOST event → flag for the Husky', 'b', 'middle', style='fill:#fff')
    s.arrow(680, 284, 680, 294, 'ax')
    return fig(s, 'The LOST rule as logic. Height evidence and camera evidence each need two independent '
                  'facts (AND), either kind will do (OR), and the detector must have lost the tree too. Then it '
                  'must hold for 3 ticks in a row. Each single fact fails in its own way; requiring them together '
                  'is what keeps the no-change gates at 0 false.')


def fig_dbscan():
    from sklearn.cluster import DBSCAN
    rng = np.random.default_rng(3)
    trunks = [(-2.0, 4.2, 0.25), (1.6, 5.4, 0.22), (3.4, 2.8, 0.28), (-4.4, 7.4, 0.3), (0.9, 8.6, 0.2)]
    pts = []
    for deg in np.arange(0, 180, 0.5):
        th = math.radians(deg)
        dx, dy = math.cos(th), math.sin(th)
        best = None
        for cx, cy, r in trunks:
            b = dx * cx + dy * cy
            c = cx * cx + cy * cy - r * r
            disc = b * b - c
            if disc >= 0:
                t = b - math.sqrt(disc)
                if t > 0 and (best is None or t < best):
                    best = t
        if best is not None and best < 12:
            t = best + rng.normal(0, 0.01)
            pts.append((t * dx, t * dy))
    for x, y in ((-5.5, 3.0), (5.2, 6.8), (-1.0, 10.4)):
        pts.append((x, y))
    P = np.array(pts)
    lab = DBSCAN(eps=0.6, min_samples=2).fit_predict(P)
    s = Svg(860, 342, 'DBSCAN on a ray-cast Husky scan: clusters at trunk fronts, stray returns as noise')
    a = Ax(s, 60, 30, 390, 260, (-6, 6), (0, 11))
    a.frame(range(-6, 7, 2), range(0, 12, 2), 'x (m)', 'y (m)')
    s.text(60, 18, 'sklearn DBSCAN(eps=0.6, min_samples=2) on one scan', 'b')
    s.poly([(a.X(0) - 7, a.Y(0) + 4), (a.X(0) + 7, a.Y(0) + 4), (a.X(0), a.Y(0) - 10)], 'fhus', closed=True)
    for cx, cy, r in trunks:
        s.circle(a.X(cx), a.Y(cy), r * a.w / 12, 'ax', fill='none', stroke_dasharray='2 2')
    pal = ['fdrs', 'fhus', 'fvios', 'fok', 'fwarn', 'facc']
    for (x, y), l in zip(P, lab):
        s.circle(a.X(x), a.Y(y), 2.6, 'fbad' if l < 0 else pal[l % len(pal)])
    biases, cents = [], []
    for l in sorted(set(lab) - {-1}):
        cx, cy = P[lab == l].mean(0)
        tc = min(trunks, key=lambda t: math.hypot(t[0] - cx, t[1] - cy))
        biases.append(math.hypot(tc[0] - cx, tc[1] - cy))
        cents.append((l, cx, cy, tc))
        s.path(f'M{a.X(cx) - 5:.1f},{a.Y(cy) - 5:.1f}l10,10M{a.X(cx) - 5:.1f},{a.Y(cy) + 5:.1f}l10,-10', 'ink',
               stroke_width=2)
    l, cx, cy, tc = max(cents, key=lambda c: c[3][2])
    ix_, iy_, isz = 480, 40, 190
    zoom = isz / 1.2
    Z = lambda x, y: (ix_ + isz / 2 + (x - tc[0]) * zoom, iy_ + isz / 2 - (y - tc[1]) * zoom)
    s.rect(ix_, iy_, isz, isz, 'fcard')
    s.circle(*Z(tc[0], tc[1]), tc[2] * zoom, 'ax', fill='none', stroke_dasharray='3 3')
    for (x, y), ll in zip(P, lab):
        if ll == l:
            s.circle(*Z(x, y), 3.5, pal[l % len(pal)])
    s.circle(*Z(tc[0], tc[1]), 3, 'ffg')
    zx, zy = Z(cx, cy)
    s.path(f'M{zx - 6:.1f},{zy - 6:.1f}l12,12M{zx - 6:.1f},{zy + 6:.1f}l12,-12', 'ink', stroke_width=2.2)
    s.arrow(*Z(tc[0], tc[1]), zx, zy, 'badl', head=6)
    s.text(ix_ + 6, iy_ + 16, 'one trunk, magnified', 'b')
    s.text(ix_ + 6, iy_ + isz - 8, f'r = {tc[2]} m: mean is {math.hypot(cx - tc[0], cy - tc[1]):.2f} m off', 'tk')
    lx = 690
    s.circle(lx + 6, 52, 4, 'fdrs'); s.text(lx + 18, 56, f'{len(set(lab) - {-1})} clusters = 5 trunks')
    s.circle(lx + 6, 76, 4, 'fbad'); s.text(lx + 18, 80, f'{int(np.sum(lab < 0))} noise (label −1)')
    s.path(f'M{lx + 1},95l10,10M{lx + 1},105l10,-10', 'ink', stroke_width=2)
    s.text(lx + 18, 104, 'cluster mean')
    s.circle(lx + 6, 124, 6, 'ax', fill='none', stroke_dasharray='2 2'); s.text(lx + 18, 128, 'true trunk')
    s.line(lx, 148, lx + 12, 148, 'badl', stroke_width=2); s.text(lx + 18, 152, 'bias')
    s.text(480, 254, 'Only the near side of a trunk returns, so the', 'mt')
    s.text(480, 270, f'mean sits in front of the centre: here {min(biases):.2f}–{max(biases):.2f} m', 'mt')
    s.text(480, 286, '(theory, half circle: 2r/π). Positions stay the', 'mt')
    s.text(480, 302, 'drone\'s; the Husky confirms "trunk there or not".', 'mt')
    return fig(s, 'Computed: a 0.5° scan from the Husky (▲) is ray-cast against five trunks, with three stray '
                  'leaf returns added, and clustered with the same DBSCAN settings as tree_mapper.py. One trunk '
                  'shadows part of another. The tight arcs on each trunk front become clusters; isolated '
                  'returns are labelled noise.')


def fig_decay():
    s = Svg(860, 260, 'Decaying hit count of one trunk cell before and after the tree is cut')
    n = 70
    rec, hits = [], 0.0
    for i in range(n):
        hits = 0.9 * hits + (1.0 if i < 30 else 0.0)
        rec.append(hits)
    lost = next(i for i in range(30, n) if rec[i] < 2)
    a = Ax(s, 70, 20, 600, 190, (0, n), (0, 11))
    a.frame(range(0, n + 1, 10), [0, 2, 4, 6, 8, 10], 'scan', 'recent hits')
    a.curve(range(n), rec, 'hu', stroke_width=2.5)
    s.line(a.X(0), a.Y(2), a.X(n), a.Y(2), 'badl', stroke_dasharray='6 4')
    s.text(a.X(n) - 4, a.Y(2) - 6, 'tree threshold 2', 'bad', 'end')
    s.line(a.X(30), a.y, a.X(30), a.y + a.h, 'ax', stroke_dasharray='4 4')
    s.text(a.X(30) + 6, a.y + 14, 'tree cut', 'b')
    s.arrow(a.X(lost), a.Y(5), a.X(lost), a.Y(rec[lost]) - 4, 'badl')
    s.text(a.X(lost) + 6, a.Y(5) - 4, f'LOST {lost - 29} scans later', 'bad b')
    s.text(a.X(1), a.Y(10.6), 'steady: → 1/(1 − 0.9) = 10', 'mt')
    lx = 700
    s.text(lx, 40, 'recent ← 0.9 · recent + hits', 'b')
    s.text(lx, 66, 'half-life ln 0.5 / ln 0.9', 'mt')
    s.text(lx, 82, '≈ 6.6 scans', 'mt')
    s.text(lx, 110, 'swept decays the same way:', 'mt')
    s.text(lx, 126, 'a cell is judged only while', 'mt')
    s.text(lx, 142, 'the robot still scans it', 'mt')
    return fig(s, f'Computed with tree_mapper\'s rule: one hit per scan builds the memory up towards 10. After '
                  f'the cut the hits stop and it decays geometrically; it crosses 2 after {lost - 29} scans and '
                  'the cell reads lost (the robot is still sweeping it). A cumulative count could never fall.')


def _jpeg_uri(img, w):
    import cv2
    h = int(img.shape[0] * w / img.shape[1])
    small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode(), h


def fig_chroma():
    import cv2
    bgr = cv2.imread(os.path.join(DATA, 'bag.jpg'))
    rgb = bgr[..., ::-1].astype(np.float32)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    red = (R >= 90) & (R >= 2 * G) & (R >= 2 * B)
    shown = bgr.copy()
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    shown[~red] = (np.stack([grey] * 3, -1)[~red] * 0.45).astype(np.uint8)
    shown[red] = (40, 40, 255)
    u1, h1 = _jpeg_uri(bgr, 210)
    u2, _ = _jpeg_uri(shown, 210)
    s = Svg(860, 352, 'A Husky camera photo, the pixels the red rule keeps, and all pixels in chromaticity space')
    s.raw(f'<image href="{u1}" x="10" y="30" width="210" height="{h1}"/>')
    s.raw(f'<image href="{u2}" x="10" y="{42 + h1}" width="210" height="{h1}"/>')
    s.text(10, 22, 'camera photo (Husky, demo run)', 'b')
    s.text(10, 58 + 2 * h1, f'kept by the red rule: {red.mean():.1%} of pixels', 'mt')
    a = Ax(s, 320, 30, 250, 250, (0, 1), (0, 1))
    a.frame([0, 0.25, 0.5, 0.75, 1], [0, 0.25, 0.5, 0.75, 1], 'r = R / (R+G+B)', 'g = G / (R+G+B)')
    tot = R + G + B + 1e-6
    r_, g_ = (R / tot).ravel(), (G / tot).ravel()
    keep = tot.ravel() > 30
    H, xe, ye = np.histogram2d(r_[keep], g_[keep], bins=40, range=[[0, 1], [0, 1]])
    for i in range(40):
        for j in range(40):
            if H[i, j] > 0:
                rc, gc = (xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2
                inside = gc <= rc / 2 and gc >= 1 - 1.5 * rc
                op = 0.15 + 0.85 * math.log10(H[i, j] + 1) / math.log10(H.max() + 1)
                s.rect(a.X(xe[i]), a.Y(ye[j + 1]), a.X(xe[i + 1]) - a.X(xe[i]) + 0.3,
                       a.Y(ye[j]) - a.Y(ye[j + 1]) + 0.3, 'fbad' if inside else 'fmt', opacity=f'{op:.2f}')
    region = [(0.5, 0.25), (2 / 3, 1 / 3), (1, 0), (2 / 3, 0)]
    s.poly([(a.X(x), a.Y(y)) for x, y in region], 'badl', closed=True, fill='none', stroke_width=2)
    s.line(a.X(0), a.Y(1), a.X(1), a.Y(0), 'ax', stroke_dasharray='3 3')
    lx = 640
    s.text(lx, 44, 'rule region (red outline):', 'b')
    s.text(lx, 64, 'R ≥ 2G  ⇔  g ≤ r / 2', 'mt')
    s.text(lx, 82, 'R ≥ 2B  ⇔  g ≥ 1 − 1.5 r', 'mt')
    s.text(lx, 100, '(plus R ≥ 90: not too dark)', 'mt')
    s.text(lx, 132, 'Chromaticity divides out', 'mt')
    s.text(lx, 148, 'brightness: a pixel in shade', 'mt')
    s.text(lx, 164, 'lands on the same spot.', 'mt')
    s.text(lx, 196, 'Grey cloud: grass, bark, sky,', 'mt')
    s.text(lx, 212, 'leaves. Red cells: the bag.', 'mt')
    s.text(lx, 228, 'Nothing else gets near the', 'mt')
    s.text(lx, 244, 'region, so a fixed rule works.', 'mt')
    return fig(s, 'Real pixels: a photo the Husky took of a bag during a demo run (top left); the pixels the '
                  'rule keeps, in red (bottom left); and every pixel of the photo placed by its colour '
                  'proportions (right, darker = more pixels). The two ratio tests cut out a triangle of colour '
                  'space that only the bag reaches.')


def fig_bagsize():
    s = Svg(860, 290, 'Expected bag pixels against distance with the 20 percent acceptance line')
    a = Ax(s, 80, 20, 560, 210, (0.5, 10), (10, 50000), ylog=True)
    a.frame([1, 2, 4, 6, 8, 10], [10, 100, 1000, 10000], 'distance d (m)', 'red points in the group')
    ry, rz = BAG_RADII[1], BAG_RADII[2]
    ds = np.linspace(0.5, 10, 200)
    whole = math.pi * ry * rz * (F_HUSKY / ds) ** 2
    need = np.maximum(0.2 * whole, 30)
    inr = (ds >= 1.2) & (ds <= 8)
    s.poly([(a.X(d), a.Y(v)) for d, v in zip(ds[inr], need[inr])] + [(a.X(8), a.y), (a.X(1.2), a.y)],
           'fok', closed=True, opacity='.15')
    a.curve(ds, whole, 'hu', stroke_width=2.5)
    a.curve(ds, need, 'badl', stroke_width=2.5, stroke_dasharray='6 4')
    for d in (1.2, 8):
        s.line(a.X(d), a.y, a.X(d), a.y + a.h, 'ax', stroke_dasharray='3 3')
    s.text(a.X(1.25) + 4, a.y + 14, 'range 1.2–8 m', 'tk')
    s.text(a.X(2.2), a.Y(math.pi * ry * rz * (F_HUSKY / 2.2) ** 2) - 8, 'a whole bag: π·0.30·0.22·(f/d)²', 'hu-t b')
    s.text(a.X(8.2), a.Y(30) - 8, 'needed: 20 % (at least 30)', 'bad')
    s.text(a.X(4.6), a.Y(20000), 'accepted', 'ok b')
    d_ex = 3.0
    need_ex = 0.2 * math.pi * ry * rz * (F_HUSKY / d_ex) ** 2
    s.circle(a.X(d_ex), a.Y(800), 6, 'fok'); s.circle(a.X(d_ex), a.Y(60), 6, 'fbad')
    lx = 670
    s.text(lx, 50, f'At 3 m a bag gives ≈ {need_ex * 5:,.0f} points;', 'mt')
    s.text(lx, 66, f'at least {need_ex:.0f} are needed.', 'mt')
    s.circle(lx + 5, 92, 5, 'fok'); s.text(lx + 16, 96, '800 points: accepted')
    s.circle(lx + 5, 116, 5, 'fbad'); s.text(lx + 16, 120, '60 points "at 3 m":')
    s.text(lx + 16, 136, 'rejected (edge pixels', 'mt')
    s.text(lx + 16, 152, 'with a branch\'s depth)', 'mt')
    return fig(s, 'Computed from the bag\'s size and the Husky camera\'s focal length (f = 207.8 px). Area on '
                  'the image falls with 1/d², so a red group must be at least a fifth of a bag\'s pixels for the '
                  'distance it claims; groups below the red line are rejected (shaded = accepted).')


def fig_yolo_arch():
    s = Svg(860, 290, 'YOLO11n: image, backbone, neck, three output grids, decoding and NMS')

    def blk(x, y, w, h, title, lines, cls='fcard'):
        s.rect(x, y, w, h, cls, rx='10')
        s.text(x + w / 2, y + 20, title, 'b', 'middle')
        for i, t in enumerate(lines):
            s.text(x + w / 2, y + 38 + i * 15, t, 'mt', 'middle')
    blk(10, 90, 110, 90, 'image', ['736 × 736', 'RGB'])
    blk(150, 70, 140, 130, 'backbone', ['convolutions', 'C3k2 blocks', 'SPPF, C2PSA', 'edges → parts'], 'fsoft')
    blk(320, 70, 140, 130, 'neck', ['FPN + PAN', 'mixes fine and', 'coarse features'], 'fsoft')
    for i, (g, st, note) in enumerate(((92, 8, 'small, far'), (46, 16, 'medium'), (23, 32, 'large, near'))):
        y = 20 + i * 88
        s.rect(500, y, 130, 70, 'fhus', rx='8', opacity='.18')
        s.text(565, y + 22, f'{g} × {g} grid', 'b', 'middle')
        s.text(565, y + 40, f'stride {st}', 'mt', 'middle')
        s.text(565, y + 56, note, 'mt', 'middle')
        s.arrow(460, 135, 498, y + 35, 'ax')
    for x in (120, 290):
        s.arrow(x, 135, x + 28, 135, 'ax')
    blk(660, 20, 190, 120, 'each of 11,109 points', ['3 class scores σ(z)', 'stump · log · bag', '4 sides × 16 bins',
                                                   '→ expected distance'])
    blk(660, 160, 190, 110, 'keep and clean', ['score ≥ 0.5', 'NMS: drop boxes with', 'IoU > 0.7 vs a better one',
                                               '→ final boxes'], 'fsoft')
    for i in range(3):
        s.arrow(630, 55 + i * 88, 658, 80, 'ax')
    s.arrow(755, 140, 755, 158, 'ax')
    s.text(10, 230, '92² + 46² + 23² = 11,109 places a box can come from', 'mt')
    s.text(10, 248, '≈ 2.6 M parameters, 6.4 GFLOPs per image; 0.7 ms on the RTX 4080', 'mt')
    return fig(s, 'One forward pass from image to boxes. The backbone turns pixels into features, the neck '
                  'combines scales, and three grids predict at every cell: class scores and the four distances '
                  'to the box sides. Small, distant stumps are found on the fine 92 × 92 grid.')


def fig_ciou():
    s = Svg(860, 280, 'CIoU geometry for a predicted and a true box, and a distribution-focal side distance')
    sc = 42
    gt = (0.6, 0.6, 4.6, 3.6)
    pr = (1.8, 1.4, 6.2, 4.0)
    ox_, oy_ = 20, 250
    def R(b):
        return ox_ + b[0] * sc, oy_ - b[3] * sc, (b[2] - b[0]) * sc, (b[3] - b[1]) * sc
    ix0, iy0, ix1, iy1 = max(gt[0], pr[0]), max(gt[1], pr[1]), min(gt[2], pr[2]), min(gt[3], pr[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    agt = (gt[2] - gt[0]) * (gt[3] - gt[1]); apr = (pr[2] - pr[0]) * (pr[3] - pr[1])
    iou = inter / (agt + apr - inter)
    en = (min(gt[0], pr[0]), min(gt[1], pr[1]), max(gt[2], pr[2]), max(gt[3], pr[3]))
    c2 = (en[2] - en[0]) ** 2 + (en[3] - en[1]) ** 2
    cg = ((gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2); cp = ((pr[0] + pr[2]) / 2, (pr[1] + pr[3]) / 2)
    rho2 = (cg[0] - cp[0]) ** 2 + (cg[1] - cp[1]) ** 2
    v = 4 / math.pi ** 2 * (math.atan((gt[2] - gt[0]) / (gt[3] - gt[1])) - math.atan((pr[2] - pr[0]) / (pr[3] - pr[1]))) ** 2
    alpha = v / ((1 - iou) + v)
    ciou = iou - rho2 / c2 - alpha * v
    x, y, w, h = R(en); s.rect(x, y, w, h, 'ax', fill='none', stroke_dasharray='5 4')
    x, y, w, h = R((ix0, iy0, ix1, iy1)); s.rect(x, y, w, h, 'fvios', opacity='.35')
    x, y, w, h = R(gt); s.rect(x, y, w, h, 'okl', fill='none', stroke_width=2.5)
    x, y, w, h = R(pr); s.rect(x, y, w, h, 'accl', fill='none', stroke_width=2.5)
    P = lambda p: (ox_ + p[0] * sc, oy_ - p[1] * sc)
    s.line(*P(cg), *P(cp), 'badl', stroke_width=2)
    s.circle(*P(cg), 4, 'fok'); s.circle(*P(cp), 4, 'facc')
    s.line(*P((en[0], en[1])), *P((en[2], en[3])), 'ax', stroke_dasharray='2 3')
    s.text(P(cg)[0] - 4, P(cg)[1] + 16, 'ρ', 'bad b', 'end')
    s.text(P((en[2], en[3]))[0] - 18, P((en[2], en[3]))[1] + 14, 'c', 'mt')
    s.text(ox_, 26, 'true box (green), predicted (blue)', 'b')
    lx = 300
    rows = [f'IoU = {iou:.3f}  (purple / union)', f'ρ²/c² = {rho2 / c2:.3f}  (centres apart)',
            f'v = {v:.4f}, α = {alpha:.3f}  (aspect)', f'CIoU = {ciou:.3f}', f'loss = 1 − CIoU = {1 - ciou:.3f}']
    for i, t in enumerate(rows):
        s.text(lx, 60 + i * 22, t, 'b' if i == 4 else '')
    a = Ax(s, 560, 40, 280, 170, (-0.5, 15.5), (0, 0.5))
    a.frame(range(0, 16, 3), [0, 0.25, 0.5], 'bin (grid cells)', 'probability')
    target = 6.4
    logits = -((np.arange(16) - 6.3) ** 2) / 1.6
    pmf = np.exp(logits) / np.exp(logits).sum()
    for i, pv in enumerate(pmf):
        s.rect(a.X(i - 0.35), a.Y(pv), a.X(i + 0.35) - a.X(i - 0.35), a.Y(0) - a.Y(pv), 'fhus', opacity='.75')
    ev = float((np.arange(16) * pmf).sum())
    s.line(a.X(ev), a.y, a.X(ev), a.y + a.h, 'accl', stroke_width=2)
    s.line(a.X(target), a.y, a.X(target), a.y + a.h, 'okl', stroke_dasharray='5 4', stroke_width=2)
    s.text(a.X(ev) + 6, a.y + 12, f'expected {ev:.2f}', 'tk')
    s.text(a.X(target) + 6, a.y + 26, f'true {target}', 'tk')
    s.text(560, 26, 'one side distance (DFL)', 'b')
    s.text(560, 262, 'DFL targets bins 6 and 7 (weights 0.6 / 0.4)', 'mt')
    return fig(s, 'Left, computed: the CIoU terms for one example. Beyond overlap (IoU), CIoU penalises the '
                  'distance between centres (ρ, relative to the enclosing box\'s diagonal c) and a wrong aspect '
                  'ratio (v), so a box learns to move even before it overlaps. Right: each box side is predicted '
                  'as a 16-bin distribution whose expectation is the distance.')


def fig_training():
    rows = list(csv.DictReader(open(os.path.join(WS, 'eval', 'yolo', 'runs', 'evidence_v1', 'results.csv'))))
    ep = [int(r['epoch']) for r in rows]
    k = {key.strip(): key for key in rows[0]}
    get = lambda name: [float(r[k[name]]) for r in rows]
    m50, m95 = get('metrics/mAP50(B)'), get('metrics/mAP50-95(B)')
    tb, vb = get('train/box_loss'), get('val/box_loss')
    tc, vc = get('train/cls_loss'), get('val/cls_loss')
    s = Svg(860, 312, 'Training curves of evidence_v1: validation mAP and train vs validation loss per epoch')
    a = Ax(s, 60, 30, 330, 210, (0, 60), (0, 1))
    a.frame(range(0, 61, 10), [0, 0.25, 0.5, 0.75, 1], 'epoch', 'validation mAP')
    a.curve(ep, m50, 'accl', stroke_width=2.5); a.curve(ep, m95, 'viol', stroke_width=2.5)
    s.line(a.X(50), a.y, a.X(50), a.y + a.h, 'ax', stroke_dasharray='4 4')
    s.text(a.X(50) - 4, a.Y(0.12), 'mosaic off', 'tk', 'end')
    s.text(a.X(60) - 2, a.Y(m50[-1]) - 8, f'mAP50 {m50[-1]:.3f}', 'b', 'end')
    s.text(a.X(60) - 2, a.Y(m95[-1]) + 18, f'mAP50-95 {m95[-1]:.3f}', 'vio-t b', 'end')
    b = Ax(s, 490, 30, 330, 210, (0, 60), (0, max(max(tb), max(vb), max(tc), max(vc)) * 1.05))
    b.frame(range(0, 61, 10), [0, 0.5, 1, 1.5, 2, 2.5], 'epoch', 'loss')
    b.curve(ep, tb, 'hu', stroke_width=2.5); b.curve(ep, vb, 'hu', stroke_width=2.5, stroke_dasharray='6 4')
    b.curve(ep, tc, 'dr', stroke_width=2.5); b.curve(ep, vc, 'dr', stroke_width=2.5, stroke_dasharray='6 4')
    s.text(b.X(60) - 2, b.Y(tb[-1]) - 8, 'box', 'hu-t b', 'end')
    s.text(b.X(60) - 2, b.Y(tc[-1]) + 16, 'class', 'dr-t b', 'end')
    s.text(490, 302, 'solid = training, dashed = validation', 'mt')
    s.text(60, 302, f'final epoch: P {get("metrics/precision(B)")[-1]:.3f}, R {get("metrics/recall(B)")[-1]:.3f}', 'mt')
    return fig(s, 'Real: the training log of the final model (eval/yolo/runs/evidence_v1/results.csv). Left: '
                  'validation mAP climbs and flattens; turning mosaic off for the last 10 epochs gives a last '
                  'small gain. Right: validation losses (dashed) fall with the training losses and do not turn '
                  'back up, so no sign of overfitting.')


def fig_pr():
    d = json.load(open(os.path.join(DATA, 'test_curves.json')))
    rec = d['recall']
    s = Svg(860, 310, 'Precision-recall curves per class on the unseen world, with the stump AP area shaded')
    a = Ax(s, 70, 20, 380, 240, (0, 1), (0, 1.02))
    a.frame([0, 0.2, 0.4, 0.6, 0.8, 1], [0, 0.2, 0.4, 0.6, 0.8, 1], 'recall', 'precision')
    cls = {'stump': 'warnl', 'log': 'accl', 'bag': 'badl'}
    st = d['precision']['stump']
    s.poly([(a.X(r), a.Y(p)) for r, p in zip(rec, st)] + [(a.X(rec[-1]), a.Y(0)), (a.X(0), a.Y(0))],
           'fwarn', closed=True, opacity='.13')
    for name, c in cls.items():
        a.curve(rec, d['precision'][name], c, stroke_width=2.5)
    lx = 500
    s.text(lx, 40, 'unseen world (simple_trees), 430 images, 1,011 objects', 'b')
    for i, name in enumerate(cls):
        y = 72 + i * 26
        s.line(lx, y - 4, lx + 26, y - 4, cls[name], stroke_width=3)
        s.text(lx + 34, y, f'{name}: AP50 = {d["ap50"][name]:.3f}')
    s.text(lx, 160, f'mAP50 = {d["map50"]:.3f}  (mean of the three)', 'b')
    s.text(lx, 180, f'mAP50-95 = {d["map"]:.3f}  (stricter boxes)', 'mt')
    s.text(lx, 214, 'AP = the area under a curve (shaded for stump).', 'mt')
    s.text(lx, 232, 'Curves stay near precision 1 until high recall:', 'mt')
    s.text(lx, 250, 'few false boxes even at low score thresholds.', 'mt')
    return fig(s, 'Real: the final model evaluated on a world it never saw in training (recomputed for this '
                  'page; it reproduces the reported mAP50 0.957 and mAP50-95 0.768). Each curve traces precision '
                  'against recall as the score threshold falls; stump recall saturates first, because small, '
                  'distant stumps are the hardest objects.')


def fig_autolabel():
    s = Svg(860, 290, 'Auto-label visibility test seen from above: projected stump points kept only where the depth image agrees')
    stump = (6.0, 0.2, 0.45)
    trunk = (3.3, 0.42, 0.3)
    sc, ox_, oy_ = 80, 40, 150
    P = lambda x, y: (ox_ + x * sc, oy_ - y * sc)

    def first_hit(dx, dy):
        best = None
        for cx, cy, r in (stump, trunk):
            b = dx * cx + dy * cy; c = cx * cx + cy * cy - r * r; disc = b * b - c
            if disc >= 0:
                t = b - math.sqrt(disc)
                if t > 0 and (best is None or t < best):
                    best = t
        return best
    s.circle(*P(stump[0], stump[1]), stump[2] * sc, 'ax', fill='none')
    s.circle(*P(trunk[0], trunk[1]), trunk[2] * sc, 'fmt', opacity='.45')
    s.text(P(trunk[0], trunk[1])[0], P(trunk[0], trunk[1] + 0.55)[1], 'tree trunk in front', 'mt', 'middle')
    s.text(P(stump[0], stump[1])[0], P(stump[0], stump[1] + 0.75)[1], 'stump', 'mt', 'middle')
    vis_u, all_u = [], []
    counts = {'vis': 0, 'occ': 0, 'back': 0}
    for k in range(32):
        ang = 2 * math.pi * k / 32
        px, py = stump[0] + stump[2] * math.cos(ang), stump[1] + stump[2] * math.sin(ang)
        d = math.hypot(px, py)
        dx, dy = px / d, py / d
        D = first_hit(dx, dy)
        visible = D is not None and abs(D - d) < 0.10 + 0.03 * d
        hit_trunk = D is not None and D < stump[0] - stump[2] - 0.05 and not visible
        u = math.atan2(py, px)
        all_u.append(u)
        if visible:
            vis_u.append(u); counts['vis'] += 1
            s.line(*P(0, 0), *P(px, py), 'okl', opacity='.25')
            s.circle(*P(px, py), 3.5, 'fok')
        else:
            counts['occ' if hit_trunk else 'back'] += 1
            if hit_trunk:
                s.line(*P(0, 0), *P(D * dx, D * dy), 'badl', opacity='.45')
                s.line(*P(D * dx, D * dy), *P(px, py), 'badl', opacity='.45', stroke_dasharray='3 3')
            s.circle(*P(px, py), 3.5, 'fbad' if hit_trunk else 'fwarn')
    s.poly([(P(0, 0)[0] - 8, P(0, 0)[1] - 7), (P(0, 0)[0] + 8, P(0, 0)[1]), (P(0, 0)[0] - 8, P(0, 0)[1] + 7)],
           'fhus', closed=True)
    s.text(P(0, 0)[0], P(0, 0)[1] + 24, 'camera', 'mt', 'middle')
    bx = 640
    s.text(bx, 30, 'image columns (u)', 'b')
    def col(u):
        return bx + 100 - math.tan(u) / math.tan(1.0472) * 100
    s.rect(bx, 50, 200, 26, 'fcard')
    lo, hi = sorted((col(min(all_u)), col(max(all_u))))
    s.rect(lo, 50, hi - lo, 26, 'fwarn', opacity='.35')
    lo2, hi2 = sorted((col(min(vis_u)), col(max(vis_u))))
    s.rect(lo2 - 2, 86, hi2 - lo2 + 4, 26, 'okl', fill='none', stroke_width=2.5)
    s.text(bx, 132, 'orange: naive projection (all points)', 'warn')
    s.text(bx, 150, 'green box: visible points ± 2 px', 'ok')
    lx = 640
    s.circle(lx + 5, 186, 4, 'fok'); s.text(lx + 16, 190, f'{counts["vis"]} visible: |D − d| < 0.10 + 0.03 d')
    s.circle(lx + 5, 210, 4, 'fbad'); s.text(lx + 16, 214, f'{counts["occ"]} behind the trunk (D ≪ d)')
    s.circle(lx + 5, 234, 4, 'fwarn'); s.text(lx + 16, 238, f'{counts["back"]} on the stump\'s back side')
    return fig(s, 'Computed from above with the labeller\'s test: 32 points on the stump\'s surface are '
                  'projected into the camera. A point counts only if the depth image shows a surface at that '
                  'point\'s own depth. Points behind the trunk (red) or on the far side (orange) are dropped, '
                  'so the label box covers what the camera actually sees, not the whole object.')


def fig_boxdepth():
    rng = np.random.default_rng(5)
    obj = rng.normal(4.0, 0.07, 1300)
    back = rng.uniform(6.2, 9.4, 700)
    grass = rng.uniform(1.8, 3.4, 40)
    d = np.concatenate([obj, back, grass])
    p25, med, mean, mn = np.percentile(d, 25), np.median(d), d.mean(), d.min()
    s = Svg(860, 280, 'Depths inside one detection box: the 25th percentile lands on the object')
    s.rect(30, 50, 210, 170, 'fsoft', rx='4')
    s.rect(30, 50, 210, 170, 'warnl', fill='none', stroke_width=2.5)
    s.rect(72, 84, 126, 102, 'ax', fill='none', stroke_dasharray='4 3')
    s.path('M70,190 Q135,95 200,190 Z', 'fwarn', opacity='.5')
    s.text(135, 40, 'YOLO box (stump)', 'b', 'middle')
    s.text(135, 238, 'inner box: 20 % trimmed per side', 'mt', 'middle')
    s.text(135, 254, 'other detections\' boxes excluded', 'mt', 'middle')
    a = Ax(s, 310, 30, 360, 190, (1.5, 9.6), (0, 1000))
    a.frame(range(2, 10), [0, 250, 500, 750, 1000], 'depth of the pixel (m)', 'pixels')
    hb, he = np.histogram(d, bins=np.arange(1.5, 9.61, 0.15))
    for c, e0 in zip(hb, he[:-1]):
        if c:
            s.rect(a.X(e0) + 0.5, a.Y(c), a.X(e0 + 0.15) - a.X(e0) - 1, a.Y(0) - a.Y(c), 'fhus', opacity='.7')
    marks = [(mn, 'min', 'ax'), (p25, '25th pct', 'okl'), (med, 'median', 'viol'), (mean, 'mean', 'badl')]
    for i, (v, lab, c) in enumerate(marks):
        s.line(a.X(v), a.y, a.X(v), a.y + a.h, c, stroke_width=2.2, stroke_dasharray='' if c == 'okl' else '5 4')

    lx = 700
    s.text(lx, 40, 'object at 4.00 m', 'b')
    for i, (v, lab, c) in enumerate(marks):
        y = 64 + i * 24
        s.line(lx, y - 4, lx + 20, y - 4, c, stroke_width=2.2, stroke_dasharray='' if c == 'okl' else '5 4')
        s.text(lx + 26, y, f'{lab} {v:.2f} m ' + ('✔' if abs(v - 4) < 0.15 else '✘'),
               'ok' if c == 'okl' else ('bad' if abs(v - 4) >= 0.15 else ''))
    s.text(lx, 170, 'median: right here, but it', 'mt')
    s.text(lx, 186, 'drifts back when background', 'mt')
    s.text(lx, 202, 'fills over half the box', 'mt')
    s.text(lx, 230, f'valid: {len(d):,} of 2,300 px', 'mt')
    s.text(lx, 246, '≥ 50 % → the box is placed', 'mt')
    return fig(s, 'An illustrative box (the distribution is simulated, the statistics are exact): most inner-box '
                  'pixels are the object at 4 m, some are background 6–9 m behind it, a few are grass in front. '
                  'The 25th percentile sits on the object; the mean is dragged backwards and the minimum is a '
                  'single noisy pixel.')


def fig_dedup():
    s = Svg(860, 330, 'Area 2 and the grown flag 5 from demo_1035, to scale, with the checks that missed and caught it')
    old, new = (5.5, 2.2, 9.5, 8.5), (6.2, 2.0, 15.2, 9.8)
    oc, nc = (7.6, 3.0), (12.5, 4.7)
    a = Ax(s, 60, 20, 420, 270, (2, 17), (0, 11))
    a.frame(range(2, 18, 2), range(0, 12, 2), 'x (m)', 'y (m)')
    def R(b, cls, **k):
        s.rect(a.X(b[0]), a.Y(b[3]), a.X(b[2]) - a.X(b[0]), a.Y(b[1]) - a.Y(b[3]), cls, **k)
    R(new, 'fvios', opacity='.18'); R(new, 'viol', fill='none', stroke_width=2)
    R(old, 'fhus', opacity='.3'); R(old, 'hu', fill='none', stroke_width=2.5)
    R((old[0] - 2, old[1] - 2, old[2] + 2, old[3] + 2), 'hu', fill='none', stroke_dasharray='5 4')
    inter = (max(old[0], new[0]), max(old[1], new[1]), min(old[2], new[2]), min(old[3], new[3]))
    R(inter, 'ink', fill='none', stroke_width=1, stroke_dasharray='2 2')
    s.circle(a.X(oc[0]), a.Y(oc[1]), 5, 'fhus'); s.circle(a.X(nc[0]), a.Y(nc[1]), 5, 'fvios')
    s.arrow(a.X(oc[0]) + 6, a.Y(oc[1]) - 2, a.X(nc[0]) - 7, a.Y(nc[1]) + 2, 'ax')
    s.text((a.X(oc[0]) + a.X(nc[0])) / 2, a.Y(3.85) - 6, '5.2 m', 'tk', 'middle')
    s.text(a.X(9.5) + 4, a.Y(10.4) + 4, 'old box + 2 m margin', 'hu-t')
    s.text(a.X(13.0), a.Y(9.2), 'flag #5', 'vio-t b', 'middle')
    s.text(a.X(7.5), a.Y(7.4), 'area 2', 'hu-t b', 'middle')
    ia = (inter[2] - inter[0]) * (inter[3] - inter[1])
    ao = (old[2] - old[0]) * (old[3] - old[1]); an = (new[2] - new[0]) * (new[3] - new[1])
    lx = 520
    rows = [('1. near a recent site? (10 m, 300 s)', 'window expired → missed', 'bad'),
            ('2. centre inside old box + 2 m?', 'x 12.5 > 11.5 → missed', 'bad'),
            (f'IoU = {ia:.1f} / {ao + an - ia:.1f}', f'= {ia / (ao + an - ia):.2f} → "new"', 'bad'),
            (f'3. overlap coef. = {ia:.1f} / min({ao:.1f}, {an:.1f})', f'= {ia / min(ao, an):.3f} ≥ 0.5 → same', 'ok')]
    for i, (q, r_, c) in enumerate(rows):
        s.text(lx, 50 + i * 54, q, 'b')
        s.text(lx, 68 + i * 54, r_, c)
    return fig(s, 'Real geometry from demo_1035, to scale. The clearing grew as the drone kept mapping; its '
                  'new centre (purple) moved 5.2 m and landed 1 m outside the old box\'s margin (dashed). IoU '
                  'divides by the union and calls it new; the overlap coefficient divides by the smaller box and '
                  'sees that 82.5 % of area 2 lies inside flag #5. The new check would have skipped the repeat.')


def fig_sweep():
    rep = json.load(open(os.path.join(WS, 'eval', 'demo', 'demo_1035', 'report.json')))
    truth = [e for e in rep['evidence'] if e.get('near') == 'oak_136']
    truth += [{'kind': 'bag', 'x': w['x'], 'y': w['y']} for w in rep['waste'] if w['near'] == 'oak_136']
    items = [i for i in rep['evidence_items'] if 3 <= i['x'] <= 14 and 0 <= i['y'] <= 11]
    loss, grown = (5.5, 2.2, 9.5, 8.5), (3.5, 0.2, 11.5, 10.5)
    wps = [(3.5, 0.2), (3.5, 10.5), (6.2, 10.5), (6.2, 0.2), (8.8, 0.2), (8.8, 10.5), (11.5, 10.5), (11.5, 0.2)]
    s = Svg(860, 340, 'Area 2 of demo_1035: the loss box, the grown search box, the 8 waypoints and the evidence')
    a = Ax(s, 60, 16, 380, 280, (2, 14), (-0.5, 11.5))
    a.frame(range(2, 15, 2), range(0, 12, 2), 'x (m)', 'y (m)')
    def R(b, cls, **k):
        s.rect(a.X(b[0]), a.Y(b[3]), a.X(b[2]) - a.X(b[0]), a.Y(b[1]) - a.Y(b[3]), cls, **k)
    R(grown, 'fsoft'); R(grown, 'ax', fill='none', stroke_dasharray='5 4')
    R(loss, 'fdrs', opacity='.22'); R(loss, 'dr', fill='none', stroke_width=2)
    s.poly([(a.X(x), a.Y(y)) for x, y in wps], 'accl', stroke_width=2.5)
    for i, (x, y) in enumerate(wps):
        s.circle(a.X(x), a.Y(y), 9, 'facc')
        s.text(a.X(x), a.Y(y) + 4, str(i + 1), 'b', 'middle', style='fill:#fff;font-size:11px')
    sym = {'stump': 'fwarn', 'log': 'facc', 'bag': 'fbad'}
    for t in truth:
        s.circle(a.X(t['x']), a.Y(t['y']), 7, 'ink', fill='none', stroke_width=2)
    for it in items:
        s.circle(a.X(it['x']), a.Y(it['y']), 4.5, sym[it['kind']])
    s.text(a.X(7.5), a.Y(5.4), 'loss box', 'dr-t b', 'middle')
    lx = 480
    s.text(lx, 40, 'From the run\'s log and report:', 'b')
    s.rect(lx, 56, 22, 14, 'fdrs', opacity='.3'); s.text(lx + 30, 67, 'loss box (5.5, 2.2)–(9.5, 8.5)')
    s.rect(lx, 82, 22, 14, 'fsoft', stroke='var(--line2)', stroke_dasharray='4 3'); s.text(lx + 30, 93, 'grown by 2 m: 8 × 10.3 m')
    s.circle(lx + 11, 115, 8, 'facc'); s.text(lx + 30, 119, '4 lanes along the 10.3 m side, 2.67 m apart')
    s.circle(lx + 11, 141, 7, 'ink', fill='none', stroke_width=2); s.text(lx + 30, 145, 'true stump, log, bag (removal test)')
    for i, (k_, c_) in enumerate(sym.items()):
        s.circle(lx + 11, 167 + i * 22, 4.5, c_)
        it = next((x for x in items if x['kind'] == k_), None)
        tr = next((x for x in truth if x['kind'] == k_), None)
        err = math.hypot(it['x'] - tr['x'], it['y'] - tr['y']) if it and tr else float('nan')
        s.text(lx + 30, 171 + i * 22, f'YOLO {k_}: {it["sightings"]} sightings, {err:.2f} m off' if it else k_)
    s.text(lx, 250, 'n = max(2, ⌈8 / 3⌉ + 1) = 4 lanes → 8 waypoints;', 'mt')
    s.text(lx, 268, 'every point is ≤ 1.34 m from a lane.', 'mt')
    s.text(lx, 286, 'Verdict: CUT ("1 stump, 1 felled log").', 'ok b')
    return fig(s, 'Real: Area 2 of today\'s run (the oak oak_136, cut). The waypoints are the ones the coordinator '
                  'logged, in order, starting at the corner nearest the Husky. Black rings are where the removal '
                  'test placed the stump, log and bag; dots are what YOLO reported.')


def fig_footprint():
    s = Svg(860, 270, 'Drone survey strips 9 m apart and the camera footprints at oak-top height')
    a = Ax(s, 60, 20, 600, 200, (-24, 24), (0, 11))
    a.frame(range(-24, 25, 6), range(0, 11, 2), 'across-track (m)', 'height (m)')
    t60 = math.tan(math.radians(60))
    cols = ['dr', 'hu', 'viol']
    fills = ['fdrs', 'fhus', 'fvios']
    for i, xc in enumerate((-9, 0, 9)):
        for side in (-1, 1):
            xe = xc + side * 10 * t60
            ze = 0.0
            if abs(xe) > 24:
                xe = 24 * side
                ze = 10 - abs(xe - xc) / t60
            s.line(a.X(xc), a.Y(10), a.X(xe), a.Y(ze), cols[i], opacity='.45')
        half = (10 - 6.2) * t60
        s.rect(a.X(xc - half), a.Y(6.2) - 5 + i * 5 - 5, a.X(xc + half) - a.X(xc - half), 5, fills[i], opacity='.75')
        s.rect(a.X(xc) - 10, a.Y(10) - 6, 20, 9, 'fsoft', rx='3')
    s.line(a.X(-24), a.Y(6.2), a.X(24), a.Y(6.2), 'okl', stroke_dasharray='5 4')
    s.line(a.X(-24), a.Y(4.6), a.X(24), a.Y(4.6), 'ax', stroke_dasharray='2 3')
    s.text(a.X(24) - 4, a.Y(6.2) - 16, 'oak tops 6.2 m', 'tk', 'end')
    s.text(a.X(24) - 4, a.Y(4.6) + 14, 'pine tops 4.6 m', 'tk', 'end')
    w = 2 * (10 - 6.2) * t60
    lx = 690
    s.text(lx, 40, 'W(z) = 2 (10 − z) tan 60°', 'b')
    s.text(lx, 66, f'oak tops: {w:.1f} m wide', '')
    s.text(lx, 84, f'pine tops: {2 * 5.4 * t60:.1f} m', '')
    s.text(lx, 102, f'ground: {20 * t60:.1f} m', '')
    s.text(lx, 132, 'strips 9 m apart:', 'b')
    s.text(lx, 150, f'{w - 9:.1f} m shared at oak tops', '')
    s.text(lx, 168, f'= {(w - 9) / w:.0%} side overlap', '')
    s.text(lx, 196, 'every crown is seen from', 'mt')
    s.text(lx, 212, 'at least two strips', 'mt')
    return fig(s, 'Computed: three survey strips 9 m apart at 10 m. The camera\'s 120° view covers a band '
                  'that narrows as the surface rises; the coloured bars are each strip\'s footprint at oak-top '
                  'height. Even there they overlap by 32 %, so canopy cells collect repeated views (the change '
                  'rule needs ≥ 3 baseline scans).')


def fig_radio():
    f = 2.4
    need = -85 + 10
    ds = np.linspace(1, 80, 400)
    curves = [
        ('Husky → base, all through trees', 'badl', lambda d: -6 + 0 + 3 - radio_model.fspl_db(d, f) - radio_model.foliage_db(d, f)),
        ('Husky → base, open ground', 'warnl', lambda d: -6 + 0 + 3 - radio_model.fspl_db(d, f)),
        ('drone → base, above the canopy', 'okl', lambda d: -6 + 2 + 3 - radio_model.fspl_db(d, f)),
    ]
    s = Svg(860, 300, 'Received power against distance for three kinds of link and the level the receiver needs')
    a = Ax(s, 70, 20, 480, 220, (0, 80), (-110, -40))
    a.frame(range(0, 81, 10), range(-110, -39, 10), 'distance (m)', 'received power (dBm)')
    s.rect(a.X(0), a.Y(need), a.w, a.Y(-110) - a.Y(need), 'fbad', opacity='.07')
    s.line(a.X(0), a.Y(need), a.X(80), a.Y(need), 'badl', stroke_dasharray='6 4', stroke_width=2)
    lx = 590
    s.text(lx, 184, 'needed: −85 dBm sensitivity', 'bad')
    s.text(lx, 200, '+ 10 dB fade margin = −75 dBm', 'bad')
    for i, (lab, cls, fn) in enumerate(curves):
        ys = [fn(d) for d in ds]
        a.curve(ds, ys, cls, stroke_width=2.5)
        reach = max((d for d, y in zip(ds, ys) if y >= need), default=0)
        s.circle(a.X(reach), a.Y(need), 5, cls.replace('l', '').replace('bad', 'fbad').replace('warn', 'fwarn').replace('ok', 'fok'))
        s.line(lx, 46 + i * 46, lx + 24, 46 + i * 46, cls, stroke_width=3)
        s.text(lx + 32, 50 + i * 46, lab)
        s.text(lx + 32, 66 + i * 46, f'reaches {reach:.0f} m', 'mt')
    s.line(lx, 168, lx + 24, 168, 'badl', stroke_dasharray='6 4', stroke_width=2)
    s.text(lx, 232, 'Trees, not distance, kill the ground', 'mt')
    s.text(lx, 248, 'link; from 10 m up only the last', 'mt')
    s.text(lx, 264, 'metres down to the Husky are foliage.', 'mt')
    return fig(s, 'Computed with radio_model.py (Friis + Weissberger, 2.4 GHz, −6 dBm, gains 0/2/3 dBi). A '
                  'link works while its curve is above the red line. Through forest the Husky reaches the base '
                  'only over about 15 m; above the canopy the drone reaches about 50 m. That gap is why the '
                  'drone is called as a relay.')


def fig_cp():
    s = Svg(860, 270, 'Clopper-Pearson 95 percent lower bound after n successes in n trials, and the rule of three')
    a = Ax(s, 70, 26, 330, 180, (1, 100), (0, 1), xlog=True)
    a.frame([1, 2, 5, 10, 20, 50, 100], [0, 0.25, 0.5, 0.75, 1], 'n trials, all succeeded', 'recall at least')
    ns = np.unique(np.round(np.logspace(0, 2, 200)).astype(int))
    a.curve(ns, 0.025 ** (1 / ns), 'accl', stroke_width=2.5)
    for n, lab in ((12, 'one run'), (20, 'drone LOST'), (60, 'YOLO, 5 runs')):
        v = 0.025 ** (1 / n)
        s.circle(a.X(n), a.Y(v), 5, 'facc')
        s.text(a.X(n), a.Y(v) - 9, str(n), 'tk b', 'middle')
        k_ = (12, 20, 60).index(n)
        s.text(a.X(9), a.Y(0.42) + k_ * 16, f'{n}/{n} → ≥ {v:.2f}  {lab}', 'tk')
    s.text(70, 18, 'lower bound = 0.025^(1/n)', 'b')
    b = Ax(s, 490, 26, 330, 180, (1, 20), (0, 3.1))
    b.frame([1, 5, 10, 15, 20], [0, 1, 2, 3], 'runs T with zero false finds', 'false finds per run, at most')
    Ts = np.linspace(1, 20, 200)
    b.curve(Ts, 3 / Ts, 'viol', stroke_width=2.5)
    s.circle(b.X(5), b.Y(0.6), 5, 'fvios')
    s.text(b.X(5) + 8, b.Y(0.6) - 8, 'our 5 runs: < 0.6 per run', 'tk')
    s.text(490, 18, 'rule of three: rate < 3 / T', 'b')
    return fig(s, 'How much a clean record proves. Left: after n successes out of n, the true success rate is '
                  'still only known to be above 0.025^(1/n) (95 %): 12/12 proves "above 0.74", not "perfect". '
                  'Right: zero false finds in T runs bounds the rate below 3/T. More runs, tighter bounds; these '
                  'runs repeat one scenario, so they show repeatability, not every forest.')


FIGS = {
    'pushbroom': (fig_pushbroom, ('replace', 'Push-broom')),
    'pinhole': (fig_pinhole, ('replace', 'Similar triangles')),
    'change': (fig_change_timeline, ('after', 'lost on a thinly-seen ("filled") baseline <span class="c">→ −40</span>\n</div>')),
    'otsu': (fig_otsu, ('replace', 'Bimodal')),
    'saddle': (fig_saddle, ('replace', 'Saddle test')),
    'itdsteps': (fig_itd_steps, ('after', '<!--ITD-STEPS-->')),
    'dbscanvsitd': (fig_dbscan_vs_itd, ('after', '<!--CLUSTER-VS-->')),
    'f1': (fig_f1, ('after', '<i>F</i><sub>1</sub> = <span class="fr"><span>2<i>PR</i></span><span><i>P</i> + <i>R</i></span></span>\n</div>')),
    'gates': (fig_gates, ('replace', 'Adaptive association gates')),
    'lost': (fig_lost_logic, ('after', 'CameraEvidence ⇔ two finished camera visits both saw through the spot (seen-canopy &lt; 0.3) ∧ the height map agrees it dropped\n</div>')),
    'dbscan': (fig_dbscan, ('replace', 'DBSCAN core')),
    'decay': (fig_decay, ('after', 'lost cell ⇔ baseline<sub>hits</sub> ≥ 2 ∧ swept ≥ 0.5 ∧ recent &lt; 2</div>')),
    'chroma': (fig_chroma, ('after', '<div class="eq">red(<i>R</i>, <i>G</i>, <i>B</i>) ⇔ <i>R</i> ≥ 90 ∧ <i>R</i> ≥ 2<i>G</i> ∧ <i>R</i> ≥ 2<i>B</i></div>')),
    'bagsize': (fig_bagsize, ('after', '<td>28 → 30 (minimum)</td></tr></table>')),
    'yoloarch': (fig_yolo_arch, ('after', "each point predicts: &nbsp; 3 class scores σ(<i>z<sub>k</sub></i>) &nbsp; and &nbsp; distances to the box's 4 sides (<i>l</i>, <i>t</i>, <i>r</i>, <i>b</i>)\n</div>")),
    'ciou': (fig_ciou, ('after', 'top 10 points per object <span class="c">s = class score, u = IoU with the object</span>\n</div>')),
    'training': (fig_training, ('after', '<td>Repeatable training.</td></tr>\n</table>')),
    'pr': (fig_pr, ('after', '<td>stump 0.990 / 0.878 / 0.931 · log 0.975 / 0.915 / 0.965 · bag 0.968 / 0.943 / 0.973</td></tr>\n</table>')),
    'autolabel': (fig_autolabel, ('after', 'label = bounding box of the visible points ± 2 px; needs ≥ 5 visible points and ≥ 6 px per side\n</div>')),
    'boxdepth': (fig_boxdepth, ('after', "<span class=\"c\">TF at the frame's timestamp</span>\n</div>")),
    'dedup': (fig_dedup, ('replace', 'Today')),
    'sweep': (fig_sweep, ('replace', 'Lawnmower sweep')),
    'footprint': (fig_footprint, ('after', 'ground: 34.6 m &nbsp;·&nbsp; pine tops (4.6 m): 18.7 m &nbsp;·&nbsp; oak tops (6.2 m): 13.2 m</div>')),
    'radio': (fig_radio, ('replace', 'Link margins')),
    'cp': (fig_cp, ('after', 'rate &lt; 3 / <i>T</i> per run (95 %)</div>')),
}


def place_markers(page):
    for name, (_, (how, key)) in FIGS.items():
        if f'<!--FIG:{name}-->' in page:
            continue
        mark = f'<!--FIG:{name}--><!--/FIG:{name}-->'
        if how == 'replace':
            pat = re.compile(r'<div class="svgbox"><svg [^>]*aria-label="' + re.escape(key) + r'[^"]*".*?</div>', re.S)
            page, n = pat.subn(lambda m_: mark, page, count=1)
        else:
            n = page.count(key)
            if n == 1:
                page = page.replace(key, key + '\n' + mark)
        if n != 1:
            raise SystemExit(f'figure {name}: anchor found {n} times')
    return page


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--page', default=os.path.join(HERE, 'mac', 'MATHS_GUIDE.html'))
    ap.add_argument('--only', nargs='*')
    a = ap.parse_args()
    page = place_markers(open(a.page).read())
    for name, (fn, _) in FIGS.items():
        if a.only and name not in a.only:
            continue
        body = fn()
        page = re.sub(f'<!--FIG:{name}-->.*?<!--/FIG:{name}-->',
                      lambda m_: f'<!--FIG:{name}-->{body}<!--/FIG:{name}-->', page, flags=re.S)
        print(f'{name:10s} {len(body) / 1024:6.1f} KB')
    open(a.page, 'w').write(page)
    print(f'wrote {a.page} ({len(page) / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
