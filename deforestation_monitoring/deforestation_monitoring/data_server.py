#!/usr/bin/env python3
"""
Data Server: Connects to ROS via rosbridge websocket, stores all data,
and serves a web dashboard. Survives page refresh — all history is preserved.

Run as a standalone ROS 2 node via the launch system (it does not itself
use rclpy — it talks to the graph through rosbridge at ws://localhost:9090).

Subscribes to the live topic set used by the system:
  /suspicious_areas, /inspection_reports, /mission_status, /survey_status,
  /baseline_status, /drone_baseline_status, /evaluation_summary,
  /canopy_change_events, /forest_change_events, change-marker arrays,
  canopy + change occupancy grids, both robot odometries, the drone
  lidar point cloud and both compressed camera streams.
"""

import json
import time
import threading
import base64
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import websocket


# ── Data store ──
store = {
    'flags': [],
    'reports': [],
    'tree_count': 0,
    'change_events': [],
    'canopy_map': None,
    'canopy_change_map': None,
    'change_map': None,
    'drone_pos': {'x': 0, 'y': 0},
    'husky_pos': {'x': 0, 'y': 0},
    'drone_points': 0,
    'change_markers': 0,
    'mission_status': '',
    'survey_status': '',
    'baseline_status': '',
    'drone_baseline_status': '',
    'evaluation': '',
    'started': time.time(),
    'drone_cam': None,   # latest compressed JPEG bytes
    'husky_cam': None,
}
store_lock = threading.Lock()


def _parse_count(text):
    """Extract an integer count from a 'label: N' status string."""
    try:
        return int(text.split(':')[1].strip().split()[0])
    except (IndexError, ValueError):
        return 0


# ── Rosbridge connection ──
def rosbridge_thread():
    """Connect to rosbridge websocket and subscribe to all relevant topics."""
    ws = None
    while True:
        try:
            ws = websocket.create_connection('ws://localhost:9090', timeout=30)

            # Subscribe to all topics (type names are rosbridge notation:
            # package/Message)
            subscriptions = {
                # Decision & communication
                '/suspicious_areas': 'deforestation_interfaces/SuspiciousArea',
                '/inspection_reports': 'std_msgs/String',
                '/mission_status': 'std_msgs/String',
                '/survey_status': 'std_msgs/String',
                # Baseline + evaluation
                '/baseline_status': 'std_msgs/String',
                '/drone_baseline_status': 'std_msgs/String',
                '/evaluation_summary': 'std_msgs/String',
                # Change events + markers
                '/canopy_change_events': 'std_msgs/String',
                '/forest_change_events': 'std_msgs/String',
                '/suspicious_markers': 'visualization_msgs/MarkerArray',
                '/tree_change_markers': 'visualization_msgs/MarkerArray',
                '/canopy_change_markers': 'visualization_msgs/MarkerArray',
                # Maps
                '/forest_canopy_map': 'nav_msgs/OccupancyGrid',
                '/canopy_change_map': 'nav_msgs/OccupancyGrid',
                '/forest_change_map': 'nav_msgs/OccupancyGrid',
                # Robots
                '/parrot1/odometry': 'nav_msgs/Odometry',
                '/husky1/odometry': 'nav_msgs/Odometry',
                '/drone_lidar_points': 'sensor_msgs/PointCloud2',
                '/parrot1/camera/compressed': 'sensor_msgs/CompressedImage',
                '/husky1/camera/compressed': 'sensor_msgs/CompressedImage',
            }
            for topic, mtype in subscriptions.items():
                ws.send(json.dumps({
                    'op': 'subscribe', 'topic': topic, 'type': mtype,
                }))

            while True:
                try:
                    msg = json.loads(ws.recv())
                except websocket.WebSocketTimeoutException:
                    # Healthy idle link: the 30 s recv timeout is expected.
                    # Reconnecting here would leak dead clients on rosbridge.
                    continue
                if msg.get('op') != 'publish':
                    continue
                topic = msg.get('topic', '')
                data = msg.get('msg', {})

                with store_lock:
                    if topic == '/suspicious_areas':
                        # SuspiciousArea message: position is at top level
                        pos = data.get('position', {})
                        px = pos.get('x', 0)
                        py = pos.get('y', 0)
                        store['flags'].append({
                            'x': px, 'y': py,
                            'time': time.time(),
                            'inspected': False,
                            'type': data.get('type', ''),
                            'confidence': data.get('confidence', 0),
                            'severity': data.get('severity', 0),
                            'area_m2': data.get('area_m2', 0),
                        })
                        if len(store['flags']) > 200:
                            store['flags'] = store['flags'][-200:]

                    elif topic == '/inspection_reports':
                        store['reports'].append({
                            'text': data.get('data', ''),
                            'time': time.time(),
                        })
                        for f in store['flags']:
                            f['inspected'] = True
                        if len(store['reports']) > 50:
                            store['reports'] = store['reports'][-50:]

                    elif topic == '/mission_status':
                        store['mission_status'] = data.get('data', '')

                    elif topic == '/survey_status':
                        store['survey_status'] = data.get('data', '')

                    elif topic == '/baseline_status':
                        store['baseline_status'] = data.get('data', '')
                        store['tree_count'] = _parse_count(data.get('data', ''))

                    elif topic == '/drone_baseline_status':
                        store['drone_baseline_status'] = data.get('data', '')

                    elif topic == '/evaluation_summary':
                        store['evaluation'] = data.get('data', '')

                    elif topic == '/canopy_change_events':
                        store['change_events'].append({
                            'text': data.get('data', ''),
                            'time': time.time(),
                        })
                        if len(store['change_events']) > 50:
                            store['change_events'] = store['change_events'][-50:]

                    elif topic == '/forest_change_events':
                        store['change_events'].append({
                            'text': data.get('data', ''),
                            'time': time.time(),
                        })
                        if len(store['change_events']) > 50:
                            store['change_events'] = store['change_events'][-50:]

                    elif topic in ('/suspicious_markers', '/tree_change_markers',
                                   '/canopy_change_markers'):
                        store['change_markers'] = len(data.get('markers', []))

                    elif topic == '/forest_canopy_map':
                        store['canopy_map'] = _grid_summary(data)

                    elif topic == '/canopy_change_map':
                        store['canopy_change_map'] = _grid_summary(data)

                    elif topic == '/forest_change_map':
                        store['change_map'] = _grid_summary(data)

                    elif topic == '/parrot1/odometry':
                        p = data.get('pose', {}).get('pose', {}).get('position', {})
                        store['drone_pos'] = {'x': p.get('x', 0), 'y': p.get('y', 0)}

                    elif topic == '/husky1/odometry':
                        p = data.get('pose', {}).get('pose', {}).get('position', {})
                        store['husky_pos'] = {'x': p.get('x', 0), 'y': p.get('y', 0)}

                    elif topic == '/drone_lidar_points':
                        store['drone_points'] = data.get('width', 0)

                    elif topic == '/parrot1/camera/compressed':
                        try:
                            store['drone_cam'] = base64.b64decode(data.get('data', ''))
                        except (ValueError, TypeError):
                            pass

                    elif topic == '/husky1/camera/compressed':
                        try:
                            store['husky_cam'] = base64.b64decode(data.get('data', ''))
                        except (ValueError, TypeError):
                            pass

        except Exception as e:
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass
                ws = None
            print(f'[rosbridge] Reconnecting in 3s: {e}')
            time.sleep(3)


def _grid_summary(data):
    """Extract the fields the dashboard needs from an OccupancyGrid dict."""
    return {
        'width': data.get('info', {}).get('width', 0),
        'height': data.get('info', {}).get('height', 0),
        'resolution': data.get('info', {}).get('resolution', 0),
        'ox': data.get('info', {}).get('origin', {}).get('position', {}).get('x', 0),
        'oy': data.get('info', {}).get('origin', {}).get('position', {}).get('y', 0),
        'data': data.get('data', []),
    }


# ── Resolve dashboard path ──
def _find_dashboard():
    """Locate the dashboard HTML file relative to the script or in standard ROS paths."""
    # First: check relative to this file (source tree layout)
    rel = os.path.join(os.path.dirname(__file__), '..', 'config', 'dashboard.html')
    if os.path.isfile(rel):
        return os.path.abspath(rel)
    # Second: check the installed share directory
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(
            get_package_share_directory('deforestation_monitoring'),
            'config', 'dashboard.html',
        )
    except ImportError:
        pass
    # Fallback: check a known ROS workspace path
    for base in ['/home/alig/41068_ws/src', os.path.expanduser('~/41068_ws/src')]:
        p = os.path.join(base, 'deforestation_monitoring', 'config', 'dashboard.html')
        if os.path.isfile(p):
            return p
    return None


HTML_PATH = _find_dashboard()


# ── HTTP server ──
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/api/data':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            with store_lock:
                # Camera frames are raw bytes served by /api/camera/* —
                # bytes are not JSON-serialisable and must not reach /api/data.
                payload = {k: v for k, v in store.items()
                           if not isinstance(v, bytes)}
                payload['uptime'] = time.time() - store['started']
            try:
                body = json.dumps(payload)
            except TypeError:
                body = json.dumps(payload, default=str)
            self.wfile.write(body.encode())
        elif path == '/api/camera/drone':
            self._serve_cached_camera('drone_cam')
        elif path == '/api/camera/husky':
            self._serve_cached_camera('husky_cam')
        else:
            # Serve dashboard HTML
            if HTML_PATH and os.path.isfile(HTML_PATH):
                try:
                    with open(HTML_PATH, 'rb') as f:
                        content = f.read()
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html')
                    self.end_headers()
                    self.wfile.write(content)
                except IOError:
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(b'Error reading dashboard')
            else:
                self.send_response(404)
                self.send_header('Content-Type', 'text/plain')
                self.end_headers()
                self.wfile.write(b'Dashboard not found. Check HTML_PATH in data_server.py')

    def _serve_cached_camera(self, key):
        """Return the latest cached camera image."""
        with store_lock:
            img = store.get(key)
        if img and len(img) > 100:
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(img)
        else:
            self.send_response(204)
            self.end_headers()

    def log_message(self, *args):
        pass  # quiet


def start_server():
    """Start the HTTP server and rosbridge listener (blocking)."""
    if HTML_PATH:
        print(f'[server] Dashboard: {HTML_PATH}')
    else:
        print('[server] WARNING: dashboard.html not found. Only /api/ endpoints available.')
    threading.Thread(target=rosbridge_thread, daemon=True).start()
    server = HTTPServer(('0.0.0.0', 8081), Handler)
    print('[server] Data server on http://0.0.0.0:8081')
    server.serve_forever()


def main(args=None):
    """Entry point for ros2 run / launch. Starts the data server."""
    start_server()


if __name__ == '__main__':
    main()
