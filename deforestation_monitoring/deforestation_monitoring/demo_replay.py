#!/usr/bin/env python3
"""Client-demo visuals from a removal-test recording.

Reads the rosbag and the removal_test JSON report of one run and renders

* ``replay.mp4``: time-lapse of the run: the drone's canopy map filling
  in, the trees it recorded (coloured by the species the camera saw, when
  the bag has ``/parrot_tree_baseline_info``) vs the simulator's true tree
  positions, the moment trees are cut, every LOST report appearing (and
  whether the height map or the camera caught it), the camera's canopy-loss
  alerts, and an inset of the drone camera when the bag has it;
* ``comparison.png``: the final frame: map + every removed tree with its
  tier, what reported it, position error and time to detect, and the score.

Scoring comes from the report (the removal test's own per-tree results and
false events), so the video agrees with the test verdict.

    ros2 run deforestation_monitoring demo_replay \\
        --bag ~/rs1_18_ws/recordings/dense_forest_new_model/bag \\
        --report ~/rs1_18_ws/recordings/dense_forest_new_model/report.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re

import numpy as np

ALERT_RE = re.compile(r'near \((-?[\d.]+),\s*(-?[\d.]+)\)')


def _grid(msg):
    g = np.array(msg.data, dtype=np.float32).reshape(msg.info.height, msg.info.width)
    extent = (msg.info.origin.position.x,
              msg.info.origin.position.x + msg.info.width * msg.info.resolution,
              msg.info.origin.position.y,
              msg.info.origin.position.y + msg.info.height * msg.info.resolution)
    return g, extent


def read_bag(path):
    """Yield (t_sec, topic, msg) for the topics the replay needs."""
    from rclpy.serialization import deserialize_message
    from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
    from nav_msgs.msg import OccupancyGrid, Odometry
    from sensor_msgs.msg import Image, PointCloud2
    from std_msgs.msg import String
    from deforestation_interfaces.msg import TreeChangeEvent
    types = {'/forest_canopy_map': OccupancyGrid, '/parrot1/odometry': Odometry,
             '/parrot_tree_positions': PointCloud2, '/parrot_tree_baseline': PointCloud2,
             '/parrot_tree_change_events': TreeChangeEvent, '/scan_coverage': String,
             '/canopy_change_events': String, '/parrot_tree_baseline_info': String,
             '/parrot_tree_change_notes': String, '/parrot1/camera/image': Image}
    r = SequentialReader()
    r.open(StorageOptions(uri=path, storage_id='sqlite3'), ConverterOptions('cdr', 'cdr'))
    while r.has_next():
        topic, data, ts = r.read_next()
        if topic in types:
            yield ts / 1e9, topic, deserialize_message(data, types[topic])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bag', required=True)
    ap.add_argument('--report', required=True, help='removal_test JSON')
    ap.add_argument('--out-dir', default=None, help='default: next to the report')
    ap.add_argument('--frame-every', type=float, default=3.0, help='bag seconds per frame')
    ap.add_argument('--fps', type=float, default=10.0)
    ap.add_argument('--title', default=None, help='default: from the world')
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    import cv2
    from ament_index_python.packages import get_package_share_directory
    from .removal_test import _cloud_xyz
    from .tree_detection import parse_tree_truth

    rep = json.load(open(os.path.expanduser(args.report)))
    out_dir = os.path.expanduser(args.out_dir or os.path.dirname(args.report))
    world = rep.get('world', 'showcase_forest')
    truth = parse_tree_truth(os.path.join(
        get_package_share_directory('41068_ignition_bringup'), 'worlds', f'{world}.sdf'))
    removed = {r[0]: (r[1], r[2]) for r in rep['removed']}
    t_removed = rep['t_removed_epoch']
    result = rep['result']
    per_tree = {t['name']: t for t in result.get('trees', [])}
    credited = {t['event_id']: t['name'] for t in per_tree.values()
                if t.get('detected') and t.get('event_id') is not None}
    false_ids = {e.get('id') for e in result.get('false_event_list', [])}
    tiered = any('tier' in t for t in per_tree.values())
    title = args.title or {
        'dense_forest': 'Dense forest (214 trees, pines under oak crowns)',
        'showcase_forest': 'Showcase forest (every tree visible from above)',
        'sparse_trees': 'Sparse forest (5 trees)'}.get(world, world)

    state = {'map': None, 'extent': None, 'dets': np.zeros((0, 3)), 'path': [],
             'events': [], 'coverage': 0.0, 'baseline': None, 'info': None,
             'by': {}, 'alerts': [], 'cam': None}
    video = None
    frame_size = (1920, 1080)
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    t0, next_frame = None, None
    tx = np.array([t[1] for t in truth])
    ty = np.array([t[2] for t in truth])
    standing = np.array([t[0] not in removed for t in truth])
    cmap = LinearSegmentedColormap.from_list(
        'canopy', ['#cdb994', '#a8c686', '#4f9a4a', '#1f5f2a', '#0c3317'])
    cmap.set_bad('#2b2b2b')                   # not yet scanned
    outline = [pe.withStroke(linewidth=3, foreground='white')]
    pine_c, oak_c = '#12b5a6', '#d98b1f'
    tier_name = {'canopy': 'canopy', 'crown_shared': 'crown-shared',
                 'crown-shared': 'crown-shared', 'understory': 'understory'}

    def tree_status(name, t):
        """(done, text) for one removed tree at bag time t."""
        r = per_tree.get(name, {})
        tier = tier_name.get(r.get('tier', ''), r.get('tier', ''))
        label = f"{name}{f' ({tier})' if tier else ''}"
        if r.get('detected') and t >= t_removed + (r.get('latency_s') or 0):
            how = state['by'].get(r.get('event_id'))
            how = ' by the camera' if how == 'camera' else ' by the height map' if how else ''
            return True, (f"✓ {label}: LOST{how}, {r.get('position_error', 0):.2f} m off, "
                          f"{r.get('latency_s', 0):.0f} s after the cut")
        alert = next((a for a in state['alerts'] if a['t'] <= t and math.hypot(
            a['x'] - removed[name][0], a['y'] - removed[name][1]) <= 3.0), None)
        if r.get('tier_ok') and not r.get('detected') and alert is not None:
            src = 'camera' if alert['camera'] else 'canopy-loss'
            return True, f"✓ {label}: {src} alert {alert['t'] - t_removed:.0f} s after the cut"
        if r.get('tier_ok') and not r.get('detected') and t >= t_final:
            return True, f"✓ {label}: not visible from above (nothing to report)"
        if t >= t_final:
            return False, f"✗ {label}: missed"
        return False, f"… {label}: waiting for the drone to re-scan"

    def render(t, final=False):
        fig.clf()
        ax = fig.add_axes([0.02, 0.05, 0.58, 0.88])
        ax.set_facecolor('#1b1f1d')
        if state['map'] is not None:
            g = np.where(state['map'] < 0, np.nan, state['map'] / 10.0)
            ax.imshow(g, origin='lower', extent=state['extent'], cmap=cmap,
                      vmin=0, vmax=7, interpolation='nearest')
        after = t >= t_removed
        ax.scatter(tx[standing], ty[standing], s=22, c='black', marker='+',
                   linewidths=1.0, label='tree (simulator truth)')
        rx = [removed[n][0] for n in removed]
        ry = [removed[n][1] for n in removed]
        ax.scatter(rx, ry, s=170 if after else 22, c='red' if after else 'black',
                   marker='x' if after else '+', linewidths=3 if after else 1.0,
                   zorder=5, label='tree cut in simulator' if after else None)
        handles = []
        if state['info'] is not None:
            # Baseline trees coloured by the species the camera saw.
            for tr in state['info']['trees']:
                col = pine_c if tr.get('species') == 'pine' else oak_c \
                    if tr.get('species') == 'oak' else '#e0e0e0'
                ax.add_patch(plt.Circle((tr['x'], tr['y']), 0.9, fill=False, color=col, lw=1.6,
                                        ls='--' if tr.get('source') == 'camera' else '-',
                                        zorder=4))
            handles += [Line2D([], [], marker='o', ls='', ms=11, mfc='none', mec=pine_c, mew=2,
                               label='pine recorded (camera colour)'),
                        Line2D([], [], marker='o', ls='', ms=11, mfc='none', mec=oak_c, mew=2,
                               label='oak recorded'),
                        Line2D([], [], ls='--', color=pine_c, lw=2,
                               label='pine only the camera found')]
        elif len(state['dets']):
            ax.scatter(state['dets'][:, 0], state['dets'][:, 1], s=90,
                       facecolors='none', edgecolors='gold', linewidths=1.5,
                       label='tree detected by drone')
        for a in state['alerts']:
            if a['camera']:
                ax.add_patch(plt.Circle((a['x'], a['y']), 1.8, fill=False, color='#ff9f0a',
                                        lw=2.5, ls=':', zorder=6))
        for e in state['events']:
            ok = e['id'] in credited
            col = '#ff3b30' if ok else '#ff9f0a'
            ax.add_patch(plt.Circle((e['x'], e['y']), 1.6, fill=False, color=col, lw=3, zorder=6))
            tag = ' (camera)' if state['by'].get(e['id']) == 'camera' else ''
            ax.annotate(f"LOST #{e['id']}{tag}", (e['x'] + 1.8, e['y'] + 1.2),
                        color='#b00020' if ok else '#8a4b00', fontsize=11,
                        weight='bold', zorder=7, path_effects=outline)
        if state['path']:
            p = np.array(state['path'][-400:])
            ax.plot(p[:, 0], p[:, 1], color='cyan', lw=1, alpha=0.6)
            ax.plot(p[-1, 0], p[-1, 1], marker='^', color='cyan', ms=14)
        ax.set_xlim(-37, 37)
        ax.set_ylim(-37, 37)
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        h0, _ = ax.get_legend_handles_labels()
        handles = h0 + handles
        if state['events']:
            handles.append(Line2D([], [], marker='o', ls='', ms=16, mfc='none',
                                  mec='#ff3b30', mew=3, label='LOST reported by drone'))
        if any(a['camera'] for a in state['alerts']):
            handles.append(Line2D([], [], marker='o', ls='', ms=16, mfc='none', mec='#ff9f0a',
                                  mew=2.5, label='camera: pine canopy gone'))
        ax.legend(handles=handles, loc='upper left', fontsize=10,
                  facecolor='white', framealpha=0.85)
        ax.set_title('Drone canopy map (LiDAR) + camera vs simulator ground truth', fontsize=16)

        show_cam = state['cam'] is not None and not final
        if show_cam:
            cax = fig.add_axes([0.625, 0.62, 0.20, 0.31])
            cax.imshow(state['cam'])
            cax.set_xticks([])
            cax.set_yticks([])
            cax.set_title('Drone camera', fontsize=12)
        tx_ = fig.add_axes([0.625, 0.05, 0.36, 0.55 if show_cam else 0.88])
        tx_.axis('off')
        elapsed = t - t0
        if state['baseline'] is None:
            phase = f"1. SURVEY: mapping the forest ({state['coverage'] * 100:.0f}% covered)"
        elif not after:
            phase = f"2. BASELINE frozen: {len(state['baseline'])} trees recorded"
        else:
            phase = f"3. MONITORING: {len(removed)} trees cut {t - t_removed:.0f} s ago"
        lines = [(title, 17, 'bold'), (f't = {elapsed:.0f} s', 12, 'normal'),
                 ('', 6, 'normal'), (phase, 14, 'bold')]
        if state['info'] is not None and not after:
            trees = state['info']['trees']
            pines = sum(1 for tr in trees if tr.get('species') == 'pine')
            cam = sum(1 for tr in trees if tr.get('source') == 'camera')
            lines += [(f'{pines} pines (camera colour), {cam} of them found only by the camera',
                       12, 'normal')]
        if after:
            done = [tree_status(n, t) for n in removed]
            wrong = sum(1 for e in state['events'] if e['id'] in false_ids)
            lines += [('', 6, 'normal'),
                      (f'Cut trees accounted for: {sum(d for d, _ in done)}/{len(removed)}',
                       13, 'bold'),
                      (f'False reports: {wrong}', 13, 'normal'), ('', 6, 'normal')]
            lines += [(text, 11, 'normal') for _, text in done]
        if final:
            if tiered:
                score = (f"canopy {result.get('canopy_found', 0)}/{result.get('canopy_trees', 0)}"
                         f" · crown-shared {result.get('crown_shared_ok', 0)}/"
                         f"{result.get('crown_shared_trees', 0)} · understory "
                         f"{result.get('understory_ok', 0)}/{result.get('understory_trees', 0)}")
            else:
                score = f"{result['true_positives']}/{result['removed']} found"
            lines += [('', 6, 'normal'),
                      (f"RESULT: {score}, {result['false_events']} false  →  "
                       f"{'PASS' if result['passed'] else 'FAIL'}", 14, 'bold')]
        y = 0.99
        for text, size, weight in lines:
            tx_.text(0.0, y, text, fontsize=size, weight=weight, va='top',
                     transform=tx_.transAxes, wrap=True)
            y -= 0.052 if size >= 13 else 0.043

    def grab():
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        return cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), frame_size)

    t_final = float('inf')
    for t, topic, msg in read_bag(os.path.expanduser(args.bag)):
        if t0 is None:
            t0, next_frame = t, t
        if topic == '/forest_canopy_map':
            state['map'], state['extent'] = _grid(msg)
        elif topic == '/parrot1/odometry':
            state['path'].append((msg.pose.pose.position.x, msg.pose.pose.position.y))
        elif topic == '/parrot_tree_positions':
            state['dets'] = _cloud_xyz(msg)
        elif topic == '/parrot_tree_baseline' and state['baseline'] is None:
            state['baseline'] = _cloud_xyz(msg)
        elif topic == '/parrot_tree_baseline_info' and state['info'] is None:
            try:
                state['info'] = json.loads(msg.data)
            except ValueError:
                pass
        elif topic == '/parrot_tree_change_notes':
            try:
                note = json.loads(msg.data)
            except ValueError:
                note = {}
            if note.get('type') == 'LOST':
                state['by'][note.get('id')] = note.get('by')
        elif topic == '/canopy_change_events' and t >= t_removed:
            for m in ALERT_RE.finditer(msg.data):
                state['alerts'].append({'x': float(m.group(1)), 'y': float(m.group(2)), 't': t,
                                        'camera': 'camera sees through' in msg.data})
        elif topic == '/parrot1/camera/image':
            if msg.encoding == 'rgb8':
                state['cam'] = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        elif topic == '/scan_coverage':
            m = re.search(r'coverage=([0-9.]+)%', msg.data)
            if m:
                state['coverage'] = float(m.group(1)) / 100.0
        elif topic == '/parrot_tree_change_events' and msg.event_type == 'LOST':
            state['events'].append({'id': msg.tree_id, 'x': msg.x, 'y': msg.y, 't': t})
        if t >= next_frame and state['map'] is not None:
            render(t)
            frame = grab()
            if video is None:
                video = cv2.VideoWriter(os.path.join(out_dir, 'replay.mp4'),
                                        cv2.VideoWriter_fourcc(*'mp4v'), args.fps, frame_size)
            video.write(frame)
            next_frame = t + args.frame_every
    t_final = t
    render(t, final=True)
    last = grab()
    for _ in range(int(args.fps * 5)):          # hold the result for 5 s
        video.write(last)
    video.release()
    cv2.imwrite(os.path.join(out_dir, 'comparison.png'), last)
    print(f"wrote {os.path.join(out_dir, 'replay.mp4')} and comparison.png "
          f"({result['true_positives']}/{result['removed']} reported LOST, "
          f"{result['false_events']} false, {'PASS' if result['passed'] else 'FAIL'})")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
