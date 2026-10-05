"""Record the dashboard (only the dashboard) to an MP4 with headless Chrome.

usage: record_dashboard.py --out demo.mp4 [--url http://localhost:8081]
                           [--size 1600x1000] [--fps 2] [--duration 1800]
                           [--stop-file /tmp/stop] [--speed 1]
                           [--finale-file /tmp/finale] [--finale-s 6]

Starts google-chrome headless with a fixed window size, opens the dashboard
and grabs a screenshot through the DevTools protocol every 1/fps seconds.
Unlike screen_recorder (the whole desktop: Gazebo, RViz, other windows), the
video shows the dashboard alone. Stops after --duration seconds, when
--stop-file appears, or on Ctrl+C; the video is always closed properly.
--speed N writes every frame once but plays at N x fps (a time-lapse).
When --finale-file appears the recording ends with the Changes view and the
report's tree table, each held for --finale-s seconds of video, then stops.
"""
import argparse
import base64
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
import urllib.request

import cv2
import numpy as np
import websocket


def devtools_page(port, timeout=20.0):
    """The websocket URL of the first page target once Chrome is up."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/json', timeout=2) as r:
                pages = [t for t in json.load(r) if t.get('type') == 'page']
            if pages:
                return pages[0]['webSocketDebuggerUrl']
        except OSError:
            pass
        time.sleep(0.5)
    raise RuntimeError('Chrome DevTools did not come up')


class Page:
    def __init__(self, ws_url):
        # Chrome rejects the default Origin header on the DevTools socket.
        self.ws = websocket.create_connection(ws_url, suppress_origin=True, timeout=30)
        self.next_id = 0

    def call(self, method, **params):
        self.next_id += 1
        self.ws.send(json.dumps({'id': self.next_id, 'method': method, 'params': params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get('id') == self.next_id:
                return msg.get('result', {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--url', default='http://localhost:8081')
    ap.add_argument('--size', default='1600x1000')
    ap.add_argument('--fps', type=float, default=2.0)
    ap.add_argument('--speed', type=float, default=1.0)
    ap.add_argument('--duration', type=float, default=3600.0)
    ap.add_argument('--stop-file')
    ap.add_argument('--port', type=int, default=9333)
    ap.add_argument('--finale-file')
    ap.add_argument('--finale-s', type=float, default=6.0)
    a = ap.parse_args()
    w, h = (int(v) for v in a.size.split('x'))

    profile = tempfile.mkdtemp(prefix='dashrec-')
    chrome = subprocess.Popen(
        ['google-chrome', '--headless=new', f'--remote-debugging-port={a.port}',
         f'--window-size={w},{h}', '--hide-scrollbars', '--no-first-run',
         '--no-default-browser-check', f'--user-data-dir={profile}', a.url],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    stop = {'now': False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    video = None
    frames = 0

    def grab(page):
        shot = page.call('Page.captureScreenshot', format='png')
        img = cv2.imdecode(np.frombuffer(base64.b64decode(shot['data']), np.uint8),
                           cv2.IMREAD_COLOR)
        if img is not None and (img.shape[1] != w or img.shape[0] != h):
            img = cv2.resize(img, (w, h))
        return img

    def hold(page, js, seconds):
        """Run js in the page, let it settle, write one still for `seconds` of video."""
        page.call('Runtime.evaluate', expression=js)
        time.sleep(1.5)
        img = grab(page)
        if img is not None and video is not None:
            for _ in range(int(seconds * a.fps * a.speed)):
                video.write(img)
    try:
        page = Page(devtools_page(a.port))
        page.call('Emulation.setDeviceMetricsOverride', width=w, height=h,
                  deviceScaleFactor=1, mobile=False)
        time.sleep(4.0)                      # let the dashboard connect and draw
        t_end = time.time() + a.duration
        period = 1.0 / a.fps
        next_t = time.time()
        while not stop['now'] and time.time() < t_end:
            if a.stop_file and os.path.exists(a.stop_file):
                break
            if a.finale_file and os.path.exists(a.finale_file) and video is not None:
                hold(page, "selectView(tabFor('changes'));renderChanges(true)", a.finale_s)
                # Scroll the report's own container, leaving room for its sticky actions bar.
                scroll_to = ("(id=>{const s=document.getElementById('report-scroll'),"
                             "e=document.getElementById(id).closest('section')||"
                             "document.getElementById(id);s.scrollTop+=e.getBoundingClientRect()"
                             ".top-s.getBoundingClientRect().top-90;})")
                hold(page, "selectView(tabFor('report'));buildReport();"
                           "document.getElementById('report-scroll').scrollTop=0", a.finale_s)
                hold(page, scroll_to + "('r-trees')", a.finale_s)
                break
            img = grab(page)
            if img is not None:
                if video is None:
                    video = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*'mp4v'),
                                            a.fps * a.speed, (w, h))
                video.write(img)
                frames += 1
            next_t += period
            time.sleep(max(0.0, next_t - time.time()))
    finally:
        if video is not None:
            video.release()
        chrome.terminate()
        try:
            chrome.wait(timeout=10)
        except subprocess.TimeoutExpired:
            chrome.kill()
        shutil.rmtree(profile, ignore_errors=True)
    print(f'wrote {a.out}: {frames} frames at {a.fps * a.speed:g} fps')


if __name__ == '__main__':
    main()
