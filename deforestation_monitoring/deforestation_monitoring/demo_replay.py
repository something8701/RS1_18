#!/usr/bin/env python3
"""Client demo visuals from a removal test recording.

Reads the rosbag and the removal_test JSON report of one run and renders:

* ``replay.mp4``: a time-lapse of the run. The drone's canopy map fills in,
  detected trees are shown next to the simulator's tree positions, and the
  cut trees and every LOST report appear as they happen.
* ``comparison.png``: the final map and a table comparing each removed tree
  (simulator ground truth) with what the system reported (position error,
  time to detect), plus any false reports.

    ros2 run deforestation_monitoring demo_replay \\
        --bag ~/rs1_18_ws/recordings/showcase_demo_4trees/bag \\
        --report ~/rs1_18_ws/recordings/showcase_demo_4trees/report.json
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np


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
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import String
    from deforestation_interfaces.msg import TreeChangeEvent
    types = {'/forest_canopy_map': OccupancyGrid, '/parrot1/odometry': Odometry,
             '/parrot_tree_positions': PointCloud2, '/parrot_tree_baseline': PointCloud2,
             '/parrot_tree_change_events': TreeChangeEvent, '/scan_coverage': String}
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
    tp, n_rem, n_false = (result['true_positives'], result['removed'],
                          result['false_events'])

    # One pass over the bag, rendering a frame every frame_every seconds
    state = {'map': None, 'extent': None, 'dets': np.zeros((0, 3)), 'path': [],
             'events': [], 'coverage': 0.0, 'baseline': None}
    video = None
    frame_size = (1920, 1080)
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    t0, next_frame = None, None
    tx = np.array([t[1] for t in truth])
    ty = np.array([t[2] for t in truth])
    standing = np.array([t[0] not in removed for t in truth])
    # Brown for ground, greens for canopy height (0-7 m)
    cmap = LinearSegmentedColormap.from_list(
        'canopy', ['#cdb994', '#a8c686', '#4f9a4a', '#1f5f2a', '#0c3317'])
    cmap.set_bad('#2b2b2b')                   # not scanned yet
    outline = [pe.withStroke(linewidth=3, foreground='white')]

    def render(t, final=False):
        fig.clf()
        ax = fig.add_axes([0.02, 0.05, 0.60, 0.88])
        ax.set_facecolor('#1b1f1d')
        if state['map'] is not None:
            g = np.where(state['map'] < 0, np.nan, state['map'] / 10.0)
            ax.imshow(g, origin='lower', extent=state['extent'], cmap=cmap,
                      vmin=0, vmax=7, interpolation='nearest')
        after = t >= t_removed
        ax.scatter(tx[standing], ty[standing], s=28, c='black', marker='+',
                   linewidths=1.2, label='tree (simulator truth)')
        rx = [removed[n][0] for n in removed]
        ry = [removed[n][1] for n in removed]
        ax.scatter(rx, ry, s=160 if after else 28, c='red' if after else 'black',
                   marker='x' if after else '+', linewidths=3 if after else 1.2,
                   zorder=5, label='tree cut in simulator' if after else None)
        if len(state['dets']):
            ax.scatter(state['dets'][:, 0], state['dets'][:, 1], s=90,
                       facecolors='none', edgecolors='gold', linewidths=1.5,
                       label='tree detected by drone')
        for e in state['events']:
            ok = e['ok']
            ax.add_patch(plt.Circle((e['x'], e['y']), 1.6, fill=False,
                                    color='#ff3b30' if ok else '#ff9f0a', lw=3, zorder=6))
            ax.annotate(f"LOST #{e['id']}", (e['x'] + 1.8, e['y'] + 1.2),
                        color='#b00020' if ok else '#8a4b00', fontsize=12,
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
        handles, _ = ax.get_legend_handles_labels()
        if state['events']:
            handles.append(Line2D([], [], marker='o', ls='', ms=16, mfc='none',
                                  mec='#ff3b30', mew=3,
                                  label='LOST reported by drone'))
        ax.legend(handles=handles, loc='upper left', fontsize=11,
                  facecolor='white', framealpha=0.85)
        ax.set_title('Drone canopy map (LiDAR) vs simulator ground truth', fontsize=16)

        tx_ = fig.add_axes([0.64, 0.05, 0.34, 0.88])
        tx_.axis('off')
        elapsed = t - t0
        if state['baseline'] is None:
            phase = f"1. SURVEY — mapping the forest ({state['coverage'] * 100:.0f}% covered)"
        elif not after:
            phase = f"2. BASELINE frozen — {len(state['baseline'])} trees recorded"
        else:
            phase = f"3. MONITORING — {len(removed)} trees cut {t - t_removed:.0f} s ago"
        lines = [('Deforestation monitoring — live demo', 20, 'bold'),
                 (f'world: {world}   t = {elapsed:.0f} s', 13, 'normal'),
                 ('', 8, 'normal'), (phase, 15, 'bold'), ('', 8, 'normal')]
        if after:
            found = sum(1 for e in state['events'] if e['ok'])
            lines += [(f'Trees cut in simulator: {len(removed)}', 14, 'normal'),
                      (f'Detected as LOST so far: {found}/{len(removed)}', 14, 'bold'),
                      (f"False reports: {sum(1 for e in state['events'] if not e['ok'])}",
                       14, 'normal'), ('', 8, 'normal')]
            for n, (x, y) in removed.items():
                ev = next((e for e in state['events'] if e.get('tree') == n), None)
                if ev:
                    lines.append((f"✓ {n}: reported {ev['err']:.2f} m from true "
                                  f"position, {ev['t'] - t_removed:.0f} s after cut", 12,
                                  'normal'))
                else:
                    lines.append((f'… {n}: waiting for the drone to re-scan', 12, 'normal'))
        if final:
            lines += [('', 8, 'normal'),
                      (f"RESULT: {tp}/{n_rem} found, {n_false} false  →  "
                       f"{'PASS' if result['passed'] else 'FAIL'}", 17, 'bold')]
        y = 0.97
        for text, size, weight in lines:
            tx_.text(0.0, y, text, fontsize=size, weight=weight, va='top',
                     transform=tx_.transAxes)
            y -= 0.045 if size >= 14 else 0.035

    def grab():
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        return cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), frame_size)

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
        elif topic == '/scan_coverage':
            import re
            m = re.search(r'coverage=([0-9.]+)%', msg.data)
            if m:
                state['coverage'] = float(m.group(1)) / 100.0
        elif topic == '/parrot_tree_change_events' and msg.event_type == 'LOST':
            near = min(removed.items(), key=lambda kv: math.hypot(
                kv[1][0] - msg.x, kv[1][1] - msg.y), default=None)
            err = math.hypot(near[1][0] - msg.x, near[1][1] - msg.y) if near else 99
            state['events'].append({'id': msg.tree_id, 'x': msg.x, 'y': msg.y, 't': t,
                                    'ok': err <= 1.5, 'err': err,
                                    'tree': near[0] if near and err <= 1.5 else None})
        if t >= next_frame and state['map'] is not None:
            render(t)
            frame = grab()
            if video is None:
                video = cv2.VideoWriter(os.path.join(out_dir, 'replay.mp4'),
                                        cv2.VideoWriter_fourcc(*'mp4v'), args.fps, frame_size)
            video.write(frame)
            next_frame = t + args.frame_every
    render(t, final=True)
    last = grab()
    for _ in range(int(args.fps * 4)):          # hold the result for 4 s
        video.write(last)
    video.release()
    cv2.imwrite(os.path.join(out_dir, 'comparison.png'), last)
    print(f"wrote {os.path.join(out_dir, 'replay.mp4')} and comparison.png "
          f"({tp}/{n_rem} found, {n_false} false)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
