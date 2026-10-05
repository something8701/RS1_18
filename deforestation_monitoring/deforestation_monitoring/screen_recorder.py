#!/usr/bin/env python3
"""Minimal X11 screen recorder (no ffmpeg needed): Pillow grab + OpenCV MP4.

Records the whole display (default :1) at ``--fps`` frames per second,
scaled to ``--width`` pixels wide, until SIGINT/SIGTERM or ``--duration``.
A timestamp overlay makes time-lapse playback readable. Used to capture the
client-showcase runs (Gazebo + RViz + dashboard) alongside the rosbags.

    ros2 run deforestation_monitoring screen_recorder --out demo.mp4 --fps 2
"""

import argparse
import signal
import time

import cv2
import numpy as np
from PIL import ImageGrab


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True, help='output .mp4')
    ap.add_argument('--display', default=':1')
    ap.add_argument('--fps', type=float, default=2.0, help='capture rate')
    ap.add_argument('--playback-fps', type=float, default=None,
                    help='video frame rate (default = capture rate; higher = time-lapse)')
    ap.add_argument('--width', type=int, default=1920)
    ap.add_argument('--duration', type=float, default=0.0, help='seconds (0 = until stopped)')
    args = ap.parse_args(argv)

    stop = {'flag': False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))

    first = ImageGrab.grab(xdisplay=args.display)
    scale = args.width / first.width
    size = (args.width, int(round(first.height * scale)) // 2 * 2)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'),
                             args.playback_fps or args.fps, size)
    t0 = time.time()
    period = 1.0 / args.fps
    frames = 0
    while not stop['flag']:
        t = time.time()
        if args.duration and t - t0 > args.duration:
            break
        img = ImageGrab.grab(xdisplay=args.display).resize(size)
        frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        stamp = time.strftime('%H:%M:%S') + f'  +{t - t0:5.0f}s'
        cv2.putText(frame, stamp, (12, size[1] - 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, stamp, (12, size[1] - 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
        frames += 1
        time.sleep(max(0.0, period - (time.time() - t)))
    writer.release()
    print(f'wrote {frames} frames ({time.time() - t0:.0f} s) to {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
