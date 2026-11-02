#!/usr/bin/env python3
# make_behaviour_tree.py: LAUNCH_LOG OUT.html: the mission as a behaviour tree (node colours = a snapshot of the run) and a …
import html
import json
import re
import sys

LOG, OUT = sys.argv[1], sys.argv[2]
SNAP = float(sys.argv[3]) if len(sys.argv) > 3 else 8 * 60 + 45

lines = open(LOG).read().splitlines()
stamp = lambda l: float(re.search(r'\[(1\d{9}\.\d+)\]', l).group(1))
t0 = next(stamp(l) for l in lines if re.search(r'\[1\d{9}\.', l) and 'removal_test' in l)
drone, husky, events, end = [], [], [], None


def close(lst, t):
    if lst and lst[-1][2] is None:
        lst[-1][2] = t


for l in lines:
    if not re.search(r'\[1\d{9}\.', l):
        continue
    t, m = max(stamp(l) - t0, 0.0), l.split(']: ', 1)[-1]
    if '[demo_patrol]' in l:
        for key, name in (('Patrol ready', 'Survey'), ('FLAG INTERRUPT', 'Fly to flag'),
                          ('Arrived at flag', 'Orbit flag'), ('Orbit complete', 'Return to survey'),
                          ('Resumed patrol', 'Survey')):
            if m.startswith(key):
                close(drone, t)
                drone.append([name, t, None])
    if '[mission_coordinator]' in l:
        if m.startswith('Mission Coordinator ready'):
            husky.append(['Wait for drone', t, None])
        elif m.startswith('Drone baseline frozen') or m.startswith('SITE INSPECTED'):
            close(husky, t)
            husky.append(['Idle', t, None])
            if m.startswith('SITE INSPECTED'):
                events.append(['Verdict', t, re.sub(r' \[image.*', '', m)])
        elif m.startswith('Dispatching Husky'):
            close(husky, t)
            husky.append(['Drive to area', t, None])
        elif m.startswith('Searching') and husky and husky[-1][0] == 'Drive to area':
            close(husky, t)
            husky.append(['Search area', t, None])
        elif m.startswith('The Husky is stuck'):
            events.append(['Stuck', t, m])
        elif m.startswith('Operator:'):
            events.append(['Operator', t, m])
    if 'tree baseline frozen' in m:
        events.append(['Baseline frozen', t, m])
    if m.startswith('[test] removed'):
        events.append(['Trees cut', t, m])
    if 'Parrot tree event: LOST' in m:
        events.append(['LOST', t, m])
    if m.startswith('EVIDENCE '):
        events.append(['YOLO find', t, m.split(', photo')[0]])
    if 'found by the Husky' in m:
        events.append(['Bag (colour)', t, m.split(', ')[0]])
    if re.match(r'\[test\] (PASS|FAIL)', m):
        end = t
        events.append(['Report', t, m.split(' —')[0]])
end = end or max(i[1] for i in drone + husky)
for lst in (drone, husky):
    close(lst, end)
    for i in lst:
        i[2] = min(i[2], end)


def mmss(s):
    return f'{int(s // 60):02d}:{int(s % 60):02d}'


def status_at(lst, name):
    return any(n == name and a <= SNAP < b for n, a, b in lst)


frozen = any(k == 'Baseline frozen' and t <= SNAP for k, t, _ in events)
drone_flag = any(status_at(drone, n) for n in ('Fly to flag', 'Orbit flag', 'Return to survey'))
h_drive, h_search = status_at(husky, 'Drive to area'), status_at(husky, 'Search area')
h_busy = h_drive or h_search
done_areas = sum(1 for k, t, _ in events if k == 'Verdict' and t <= SNAP)

S, R, F, I = 'success', 'running', 'failure', 'inactive'
FILL = {S: '#34c759', R: '#5ac8fa', F: '#ff453a', I: 'var(--card)'}
INK = {S: '#06210d', R: '#032532', F: '#2b0303', I: 'var(--fg)'}
nodes, edges = {}, []


def N(key, kind, x, y, label, st, parent=None, note=''):
    nodes[key] = (kind, x, y, label, st, note)
    if parent:
        edges.append((parent, key))


N('root', '?', 760, 46, '', R)
N('s1', '→', 110, 140, '', F, 'root')
N('s1c', 'C', 70, 230, 'Sim clock|stalled?', F, 's1')
N('s1a', 'A', 170, 230, 'Warn +|/sim_stalled', I, 's1')
N('s2', '→', 345, 140, '', F, 'root')
N('s2c', 'C', 268, 230, 'Husky|stuck?', F, 's2')
N('s2a', 'A', 360, 230, 'Reverse|1.2 m', I, 's2')
N('s2b', 'A', 450, 230, 'Mark|hazard', I, 's2')
N('s3', '→', 575, 140, '', F, 'root')
N('s3c', 'C', 545, 230, 'Sweep goal|timed out?', F, 's3')
N('s3a', 'A', 640, 230, 'Give up,|requeue', I, 's3')
N('m', '⇉', 760, 345, '', R, 'root')
N('d', '→', 380, 440, '', R, 'm')
N('db', '?', 165, 525, '', S if frozen else R, 'd')
N('dbc', 'C', 62, 630, 'Baseline|frozen?', S if frozen else F, 'db')
N('dbs', '→', 245, 630, '', I if frozen else R, 'db')
N('db1', 'A', 70, 745, 'Survey +|canopy map', I if frozen else R, 'dbs')
N('db2', 'A', 170, 745, 'Detect trees|(ITD+camera)', I if frozen else R, 'dbs')
N('db3', 'C', 268, 745, '≥90% cover,|stable?', I if frozen else F, 'dbs')
N('db4', 'A', 362, 745, 'Freeze|baseline', I, 'dbs')
N('dw', '?', 590, 525, '', R if frozen else I, 'd')
N('dwf', '→', 480, 630, '', (R if drone_flag else F) if frozen else I, 'dw')
N('dwfc', 'C', 455, 745, 'Flag|received?', (S if drone_flag else F) if frozen else I, 'dwf')
N('dwfa', 'A', 548, 745, 'Fly to flag,|orbit, return', R if drone_flag else I, 'dwf')
N('dwr', '→', 615, 630, '', F if frozen else I, 'dw')
N('dwrc', 'C', 645, 745, 'Relay|needed?', F if frozen else I, 'dwr')
N('dws', 'A', 700, 630, 'Re-survey,|LOST rules', (I if drone_flag else R) if frozen else I, 'dw')
N('h', '→', 1110, 440, '', R, 'm')
N('hc', 'C', 830, 525, 'Baseline|frozen?', S if frozen else F, 'h')
N('hw', '?', 1150, 525, '', R if frozen else I, 'h')
N('hwc', 'C', 905, 630, 'No area|waiting?', (F if h_busy else S) if frozen else I, 'hw')
N('hi', '→', 1215, 630, '', (R if h_busy else I) if frozen else I, 'hw')
N('h1', '?', 1000, 745, '', S if h_busy else I, 'hi',
  'ask on → wait for|operator, else top|priority (sev×conf÷dist)')
N('h2', 'A', 1110, 745, 'Drive to area|(Nav2 BT)', (S if h_search else R) if h_busy else I, 'hi',
  'fails twice →|UNREACHABLE')
N('h3', 'A', 1220, 745, 'Sweep|(lawnmower)', R if h_search else I, 'hi')
N('h4', 'A', 1330, 745, 'Detect: YOLO,|colour, trunks', R if h_busy else I, 'hi',
  'runs the whole|time')
N('h5', 'A', 1440, 745, 'Verdict +|report', I, 'hi', 'stump/log seen|→ CUT')


def node_svg(key):
    kind, x, y, label, st, note = nodes[key]
    f, ink = FILL[st], INK[st]
    out = []
    if kind == 'C':
        out.append(f'<circle cx="{x}" cy="{y}" r="40" fill="{f}" stroke="var(--line2)" stroke-width="1.5"/>')
    elif kind == 'A':
        out.append(f'<rect x="{x - 46}" y="{y - 24}" width="92" height="48" rx="6" fill="{f}" '
                   f'stroke="var(--line2)" stroke-width="1.5"/>')
    else:
        out.append(f'<rect x="{x - 19}" y="{y - 19}" width="38" height="38" rx="4" fill="{f}" '
                   f'stroke="var(--line2)" stroke-width="1.5"/>')
        out.append(f'<text x="{x}" y="{y + 6}" text-anchor="middle" font-size="17" font-weight="700" '
                   f'fill="{ink}">{html.escape(kind)}</text>')
    parts = label.split('|') if label else []
    for i, p in enumerate(parts):
        dy = (i - (len(parts) - 1) / 2) * 13 + 4
        out.append(f'<text x="{x}" y="{y + dy:.0f}" text-anchor="middle" font-size="11.5" '
                   f'fill="{ink}">{html.escape(p)}</text>')
    if note:
        for i, p in enumerate(note.split('|')):
            out.append(f'<text x="{x}" y="{y + 44 + i * 12}" text-anchor="middle" font-size="10" '
                       f'fill="var(--mute)">{html.escape(p)}</text>')
    return ''.join(out)


def edge_svg(a, b):
    _, x1, y1, *_ = nodes[a]
    kb, x2, y2, *_ = nodes[b]
    y1 += 19
    y2 -= 40 if kb == 'C' else 24 if kb == 'A' else 19
    my = (y1 + y2) / 2
    return (f'<path d="M{x1},{y1} C{x1},{my} {x2},{my} {x2},{y2}" fill="none" '
            f'stroke="var(--mute)" stroke-width="1.3" marker-end="url(#bt-ar)"/>')


def panel(x, y, w, h, title, color='var(--fg)'):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="none" stroke="var(--line)" '
            f'stroke-width="1.5"/><text x="{x + 14}" y="{y + 26}" font-size="19" font-weight="600" '
            f'fill="{color}">{title}</text>')


legend = ['<g transform="translate(1000,18)">',
          '<rect width="520" height="104" rx="10" fill="var(--card)" stroke="var(--line)"/>',
          '<text x="14" y="22" font-size="14" font-weight="600" fill="var(--fg)">Legend</text>']
lx = 14
for kind, name in (('C', 'Condition'), ('A', 'Action'), ('?', 'Fallback'), ('→', 'Sequence'), ('⇉', 'Parallel')):
    if kind == 'C':
        legend.append(f'<circle cx="{lx + 14}" cy="44" r="12" fill="var(--card)" stroke="var(--line2)"/>')
    elif kind == 'A':
        legend.append(f'<rect x="{lx}" y="34" width="28" height="20" rx="3" fill="var(--card)" stroke="var(--line2)"/>')
    else:
        legend.append(f'<rect x="{lx + 4}" y="34" width="20" height="20" rx="3" fill="var(--card)" stroke="var(--line2)"/>'
                      f'<text x="{lx + 14}" y="49" text-anchor="middle" font-size="13" font-weight="700" fill="var(--fg)">{kind}</text>')
    legend.append(f'<text x="{lx + 34}" y="49" font-size="12" fill="var(--fg)">{name}</text>')
    lx += 102
for i, (st, name) in enumerate(((S, 'Success'), (R, 'Running'), (F, 'Failure / false'), (I, 'Inactive'))):
    legend.append(f'<rect x="{14 + i * 125}" y="68" width="116" height="24" rx="4" fill="{FILL[st]}" stroke="var(--line2)"/>'
                  f'<text x="{72 + i * 125}" y="84" text-anchor="middle" font-size="12" fill="{INK[st]}">{name}</text>')
legend.append('</g>')

tree_svg = ('<svg viewBox="0 0 1540 850" width="100%" style="min-width:1100px" role="img" '
            'aria-label="Mission behaviour tree">'
            '<defs><marker id="bt-ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" '
            'orient="auto"><path d="M0,0L10,5L0,10z" fill="var(--mute)"/></marker></defs>'
            + panel(14, 96, 700, 220, '')
            + '<text x="700" y="304" text-anchor="end" font-size="19" font-weight="600" fill="var(--fg)">Safety (checked first)</text>'
            + panel(14, 390, 735, 440, 'Drone: survey &amp; watch', 'var(--drone)')
            + panel(765, 390, 760, 440, 'Husky: inspect areas', 'var(--husky)')
            + ''.join(edge_svg(a, b) for a, b in edges)
            + ''.join(node_svg(k) for k in nodes) + ''.join(legend)
            + f'<text x="1520" y="148" text-anchor="end" font-size="12" fill="var(--mute)">'
              f'snapshot at {mmss(SNAP)} of the run</text></svg>')

ROWS = [('Report', None), ('Verdict + report', None), ('Search area', 'h'), ('Drive to area', 'h'),
        ('Idle', 'h'), ('Wait for drone', 'h'), ('Return to survey', 'd'), ('Orbit flag', 'd'),
        ('Fly to flag', 'd'), ('Survey', 'd')]
COL = {'Survey': '#ff9f0a', 'Fly to flag': '#ff6b00', 'Orbit flag': '#d6409f', 'Return to survey': '#bf5af2',
       'Wait for drone': '#8e8e93', 'Idle': '#c7c7cc', 'Drive to area': '#0a84ff', 'Search area': '#30d158'}
X0, X1, Y0, RH = 170, 1490, 30, 30
sx = lambda t: X0 + (X1 - X0) * t / end
rows = [r for r in ROWS if r[1]]
out = [f'<svg viewBox="0 0 1520 {Y0 + RH * (len(rows) + 3) + 70}" width="100%" style="min-width:1000px" '
       'role="img" aria-label="Behaviour timeline">']
for i, (name, who) in enumerate(rows):
    y = Y0 + 50 + i * RH
    out.append(f'<text x="{X0 - 10}" y="{y + 19}" text-anchor="end" font-size="14" '
               f'fill="{"var(--husky)" if who == "h" else "var(--drone)"}">{name}</text>')
    out.append(f'<line x1="{X0}" y1="{y + RH}" x2="{X1}" y2="{y + RH}" stroke="var(--line)" stroke-width="0.6"/>')
    for n, a, b in (husky if who == 'h' else drone):
        if n == name:
            out.append(f'<rect x="{sx(a):.1f}" y="{y + 4}" width="{max(sx(b) - sx(a), 1.5):.1f}" height="{RH - 8}" '
                       f'rx="3" fill="{COL[name]}"><title>{name} {mmss(a)}–{mmss(b)}</title></rect>')
ey = Y0 + 50 + len(rows) * RH + 10
out.append(f'<text x="{X0 - 10}" y="{ey + 16}" text-anchor="end" font-size="14" fill="var(--fg)">Events</text>')
MARK = {'Baseline frozen': ('#34c759', 'F'), 'Trees cut': ('#ff453a', '✂'), 'LOST': ('#ff453a', 'L'),
        'Verdict': ('#30d158', '✓'), 'YOLO find': ('#5e5ce6', '·'), 'Bag (colour)': ('#ff375f', 'B'),
        'Stuck': ('#ff9f0a', '!'), 'Operator': ('#0a84ff', 'O'), 'Report': ('#34c759', '★')}
for k, t, m in events:
    if k not in MARK or t > end + 1:
        continue
    c, g = MARK[k]
    if k == 'YOLO find':
        out.append(f'<circle cx="{sx(t):.1f}" cy="{ey + 32}" r="3" fill="{c}"><title>{mmss(t)} {html.escape(m)}</title></circle>')
    else:
        out.append(f'<g><title>{mmss(t)} {html.escape(m)}</title><circle cx="{sx(t):.1f}" cy="{ey + 10}" r="9" '
                   f'fill="{c}"/><text x="{sx(t):.1f}" y="{ey + 14}" text-anchor="middle" font-size="11" '
                   f'font-weight="700" fill="#fff">{g}</text></g>')
ay = ey + 48
out.append(f'<line x1="{X0}" y1="{ay}" x2="{X1}" y2="{ay}" stroke="var(--fg)"/>')
for mnt in range(0, int(end // 60) + 1):
    x = sx(mnt * 60)
    out.append(f'<line x1="{x:.1f}" y1="{ay}" x2="{x:.1f}" y2="{ay + 6}" stroke="var(--fg)"/>'
               f'<text x="{x:.1f}" y="{ay + 22}" text-anchor="middle" font-size="12" fill="var(--fg)">{mnt}</text>')
out.append(f'<text x="{(X0 + X1) / 2}" y="{ay + 44}" text-anchor="middle" font-size="14" fill="var(--fg)">time (min)</text>')
out.append(f'<line x1="{sx(SNAP):.1f}" y1="{Y0 + 40}" x2="{sx(SNAP):.1f}" y2="{ay}" stroke="var(--mute)" '
           f'stroke-dasharray="4 4"/><text x="{sx(SNAP):.1f}" y="{Y0 + 34}" text-anchor="middle" font-size="12" '
           f'fill="var(--mute)">tree snapshot {mmss(SNAP)}</text>')
leg = [('F', 'baseline frozen'), ('✂', 'trees cut'), ('L', 'LOST'), ('B', 'bag (colour)'), ('·', 'YOLO find'),
       ('✓', 'area verdict'), ('★', 'report')]
out.append(f'<text x="{X0}" y="{Y0 + 8}" font-size="12" fill="var(--mute)">' + ' · '.join(f'{g} {n}' for g, n in leg) + '</text>')
out.append('</svg>')
timeline_svg = ''.join(out)

counts = {k: sum(1 for e in events if e[0] == k) for k in ('LOST', 'YOLO find', 'Bag (colour)', 'Verdict')}
page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Mission Behaviour Tree</title>
<style>
:root{{--bg:#fbfbfd;--fg:#1d1d1f;--mute:#6e6e73;--card:#fff;--line:#e3e3e8;--line2:#8e8e93;--drone:#e8710a;--husky:#0a9ec7}}
@media (prefers-color-scheme:dark){{:root{{--bg:#000;--fg:#f5f5f7;--mute:#a1a1a6;--card:#1c1c1e;--line:#38383a;--line2:#636366;--drone:#ff9f0a;--husky:#64d2ff}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 -apple-system,BlinkMacSystemFont,Helvetica,Arial,sans-serif}}
main{{max-width:1560px;margin:0 auto;padding:28px 16px 60px}}
h1{{font-size:32px;margin:0 0 6px;letter-spacing:-.02em}} h2{{font-size:22px;margin:34px 0 8px}}
.sub{{color:var(--mute);margin:0 0 14px}} .box{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:10px;overflow-x:auto}}
svg text{{font-family:-apple-system,Helvetica,Arial,sans-serif}}
table{{border-collapse:collapse;font-size:15px;margin-top:8px}} td,th{{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}
th{{color:var(--mute)}}
</style></head><body><main>
<h1>Mission behaviour tree</h1>
<p class="sub">The drone + Husky mission as a behaviour tree. Node colours show the state at {mmss(SNAP)} of the
dry-run ({html.escape(LOG.split('/')[-1])}): the baseline is frozen, the drone is re-surveying, and the Husky is
{'searching' if h_search else 'driving to' if h_drive else 'between'} area {done_areas + 1}. The timeline below is read from the same run's log.</p>
<div class="box">{tree_svg}</div>
<h2>How to read it</h2>
<table>
<tr><th>Node</th><th>Ticks its children…</th><th>Returns</th></tr>
<tr><td><b>?</b> Fallback</td><td>left to right until one does not fail</td><td>success/running of the first child that does not fail</td></tr>
<tr><td><b>→</b> Sequence</td><td>left to right while each succeeds</td><td>failure as soon as one fails; success when all succeed</td></tr>
<tr><td><b>⇉</b> Parallel</td><td>all at once (the drone and the Husky run together)</td><td>running while either runs</td></tr>
<tr><td>○ Condition</td><td>a check, instant</td><td>success (true) or failure (false)</td></tr>
<tr><td>▭ Action</td><td>does something over time</td><td>running, then success or failure</td></tr>
</table>
<p class="sub">Every tick the root first checks the safety branch: each → fails at its condition while nothing is wrong
(red), so the fallback moves on to the mission. If the Husky gets stuck, "Husky stuck?" turns true and the escape runs
before anything else. Under the mission, the drone fallback skips "build the baseline" once "Baseline frozen?" is true and
watches for change instead; the Husky sequence cannot pass "Baseline frozen?" before that, which is why it waits.
<b>Honest note:</b> the code is two state machines (mission_coordinator, demo_patrol) and the tracker's freeze rules; this
is their behaviour-tree view. Nav2's navigation inside "Drive to area" is a real behaviour tree.</p>
<h2>What each robot did, when</h2>
<p class="sub">{mmss(end)} run · {counts['LOST']} LOST · {counts['Bag (colour)']} bags found by colour · {counts['YOLO find']} YOLO finds · {counts['Verdict']} areas with a verdict. Hover a bar or marker for details.</p>
<div class="box">{timeline_svg}</div>
</main></body></html>'''
open(OUT, 'w').write(page)
STUDY = OUT.rsplit('/', 1)[0] + '/STUDY_GUIDE.html'
try:
    body = page.split('<main>', 1)[1].split('</main>', 1)[0]
    body = body.replace('<h1>Mission behaviour tree</h1>', '<h2 id="bt">Behaviour tree &amp; timeline</h2>') \
               .replace('<h2>', '<h3>').replace('</h2>', '</h3>').replace('<h3 id="bt">', '<h2 id="bt">') \
               .replace('class="box"', 'class="svgbox"')
    body = body.replace('Behaviour tree &amp; timeline</h3>', 'Behaviour tree &amp; timeline</h2>')
    body = body.replace('</h2>', '</h2><p class="sub"><a href="BEHAVIOUR_TREE.html">Open it full size</a> '
                        '(also <code>demo bt</code>); scroll sideways in the boxes below.</p>', 1)
    guide = open(STUDY).read()
    start, stop = '<!--BT-START-->', '<!--BT-END-->'
    if start in guide:
        guide = guide.split(start)[0] + start + body + stop + guide.split(stop)[1]
    else:
        guide = guide.replace('<!-- ============================================================ -->\n<h2 id="mapping">',
                              start + body + stop + '\n<!-- ============================================================ -->\n<h2 id="mapping">', 1)
        guide = guide.replace('<a href="#timeline">One run, minute by minute</a>',
                              '<a href="#timeline">One run, minute by minute</a>\n<a href="#bt">Behaviour tree &amp; timeline</a>', 1)
        guide = guide.replace('--soft:#f5f5f7}', '--soft:#f5f5f7;--line2:#8e8e93}', 1) \
                     .replace('--soft:#141416}}', '--soft:#141416;--line2:#636366}}', 1)
    open(STUDY, 'w').write(guide)
    print('updated', STUDY)
except FileNotFoundError:
    pass
print(f'wrote {OUT}: {len(nodes)} nodes, {len(drone)} drone and {len(husky)} Husky intervals, {len(events)} events')
