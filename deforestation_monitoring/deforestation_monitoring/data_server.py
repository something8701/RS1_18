#!/usr/bin/env python3
# data_server.py: Connects to ROS via rosbridge websocket, stores all data, and serves a web dashboard
import json
import time
import threading
import base64
import os
import struct
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import websocket


store = {
    'flags': [],
    'reports': [],
    'tree_count': 0,
    'tree_events': [],
    'tree_status': {
        'baseline_count': 0,
        'current_count': 0,
        'lost_count': 0,
        'gained_count': 0,
    },
    'parrot_tree_events': [],
    'parrot_tree_status': {
        'baseline_count': 0,
        'current_count': 0,
        'lost_count': 0,
        'gained_count': 0,
    },
    'parrot_tree_positions': [],
    'baseline_info': None,
    'change_notes': [],
    'species_map': None,
    'change_events': [],
    'canopy_map': None,
    'canopy_change_map': None,
    'change_map': None,
    'drone_pos': {'x': 0, 'y': 0},
    'husky_pos': None,
    'drone_points': 0,
    'change_markers': 0,
    'mission_status': '',
    'survey_status': '',
    'scan_coverage': '',
    'baseline_status': '',
    'drone_baseline_status': '',
    'evaluation': '',
    'started': time.time(),
    'drone_cam': None,
    'husky_cam': None,
}
store_lock = threading.Lock()


def _parse_count(text):
    try:
        return int(text.split(':')[1].strip().split()[0])
    except (IndexError, ValueError):
        return 0


def rosbridge_thread():
    ws = None
    while True:
        try:
            ws = websocket.create_connection('ws://localhost:9090', timeout=30)

            subscriptions = {
                '/suspicious_areas': 'deforestation_interfaces/SuspiciousArea',
                '/inspection_reports': 'std_msgs/String',
                '/mission_status': 'std_msgs/String',
                '/survey_status': 'std_msgs/String',
                '/baseline_status': 'std_msgs/String',
                '/drone_baseline_status': 'std_msgs/String',
                '/scan_coverage': 'std_msgs/String',
                '/evaluation_summary': 'std_msgs/String',
                '/tree_change_events': 'deforestation_interfaces/TreeChangeEvent',
                '/tree_status': 'deforestation_interfaces/TreeStatus',
                '/parrot_tree_change_events': 'deforestation_interfaces/TreeChangeEvent',
                '/parrot_tree_status': 'deforestation_interfaces/TreeStatus',
                '/parrot_tree_positions': 'sensor_msgs/PointCloud2',
                '/parrot_tree_baseline_info': 'std_msgs/String',
                '/parrot_tree_change_notes': 'std_msgs/String',
                '/forest_species_map': 'nav_msgs/OccupancyGrid',
                '/canopy_change_events': 'std_msgs/String',
                '/forest_change_events': 'std_msgs/String',
                '/suspicious_markers': 'visualization_msgs/MarkerArray',
                '/tree_change_markers': 'visualization_msgs/MarkerArray',
                '/canopy_change_markers': 'visualization_msgs/MarkerArray',
                '/forest_canopy_map': 'nav_msgs/OccupancyGrid',
                '/canopy_change_map': 'nav_msgs/OccupancyGrid',
                '/forest_change_map': 'nav_msgs/OccupancyGrid',
                '/parrot1/odometry': 'nav_msgs/Odometry',
                '/husky1/odometry': 'nav_msgs/Odometry',
                '/drone_lidar_points': 'sensor_msgs/PointCloud2',
                '/parrot1/camera/compressed': 'sensor_msgs/CompressedImage',
                '/husky1/camera/compressed': 'sensor_msgs/CompressedImage',
                '/husky1/yolo/compressed': 'sensor_msgs/CompressedImage',
            }
            for topic, mtype in subscriptions.items():
                ws.send(json.dumps({
                    'op': 'subscribe', 'topic': topic, 'type': mtype,
                }))

            while True:
                try:
                    msg = json.loads(ws.recv())
                except websocket.WebSocketTimeoutException:
                    continue
                if msg.get('op') != 'publish':
                    continue
                topic = msg.get('topic', '')
                data = msg.get('msg', {})

                with store_lock:
                    if topic == '/suspicious_areas':
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

                    elif topic == '/scan_coverage':
                        store['scan_coverage'] = data.get('data', '')

                    elif topic == '/tree_change_events':
                        store['tree_events'].append({
                            'event_type': data.get('event_type', ''),
                            'tree_id': data.get('tree_id', 0),
                            'x': data.get('x', 0.0),
                            'y': data.get('y', 0.0),
                            'area_m2': data.get('area_m2', 0.0),
                            'time': time.time(),
                        })
                        if len(store['tree_events']) > 100:
                            store['tree_events'] = store['tree_events'][-100:]

                    elif topic == '/tree_status':
                        store['tree_status'] = {
                            'baseline_count': data.get('baseline_count', 0),
                            'current_count': data.get('current_count', 0),
                            'lost_count': data.get('lost_count', 0),
                            'gained_count': data.get('gained_count', 0),
                        }
                        store['tree_count'] = data.get('current_count', 0)

                    elif topic == '/parrot_tree_change_events':
                        store['parrot_tree_events'].append({
                            'event_type': data.get('event_type', ''),
                            'tree_id': data.get('tree_id', 0),
                            'x': data.get('x', 0.0),
                            'y': data.get('y', 0.0),
                            'area_m2': data.get('area_m2', 0.0),
                            'time': time.time(),
                        })
                        if len(store['parrot_tree_events']) > 100:
                            store['parrot_tree_events'] = store['parrot_tree_events'][-100:]

                    elif topic == '/parrot_tree_status':
                        store['parrot_tree_status'] = {
                            'baseline_count': data.get('baseline_count', 0),
                            'current_count': data.get('current_count', 0),
                            'lost_count': data.get('lost_count', 0),
                            'gained_count': data.get('gained_count', 0),
                        }
                        store['tree_count'] = data.get('current_count', 0)

                    elif topic == '/parrot_tree_positions':
                        positions = []
                        try:
                            raw = base64.b64decode(data.get('data', ''))
                            width = data.get('width', 0)
                            step = data.get('point_step', 12)
                            for index in range(width):
                                offset = index * step
                                x, y = struct.unpack_from('<ff', raw, offset)
                                positions.append({'x': float(x), 'y': float(y)})
                        except Exception:
                            positions = []
                        store['parrot_tree_positions'] = positions
                        try:
                            out_dir = '/tmp/deforestation_eval'
                            os.makedirs(out_dir, exist_ok=True)
                            with open(os.path.join(out_dir, 'parrot_tree_positions.json'), 'w') as f:
                                json.dump(positions, f, indent=2)
                        except OSError:
                            pass

                    elif topic == '/parrot_tree_baseline_info':
                        try:
                            info = json.loads(data.get('data', ''))
                        except ValueError:
                            info = None
                        if info is not None and info != store['baseline_info']:
                            store['baseline_info'] = info
                            store['change_notes'] = []

                    elif topic == '/parrot_tree_change_notes':
                        try:
                            note = json.loads(data.get('data', ''))
                        except ValueError:
                            note = None
                        if isinstance(note, dict):
                            note['time'] = time.time()
                            store['change_notes'].append(note)
                            store['change_notes'] = store['change_notes'][-200:]

                    elif topic == '/forest_species_map':
                        store['species_map'] = _grid_summary(data)

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
                        ev = store.setdefault('husky_change_events', [])
                        ev.append({'text': data.get('data', ''), 'time': time.time()})
                        if len(ev) > 50:
                            store['husky_change_events'] = ev[-50:]

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

                    elif topic == '/husky1/yolo/compressed':
                        try:
                            store['husky_yolo'] = base64.b64decode(data.get('data', ''))
                            store['husky_yolo_t'] = time.time()
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
    return {
        'width': data.get('info', {}).get('width', 0),
        'height': data.get('info', {}).get('height', 0),
        'resolution': data.get('info', {}).get('resolution', 0),
        'ox': data.get('info', {}).get('origin', {}).get('position', {}).get('x', 0),
        'oy': data.get('info', {}).get('origin', {}).get('position', {}).get('y', 0),
        'data': data.get('data', []),
    }


def _find_dashboard():
    rel = os.path.join(os.path.dirname(__file__), '..', 'config', 'dashboard.html')
    if os.path.isfile(rel):
        return os.path.abspath(rel)
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(
            get_package_share_directory('deforestation_monitoring'),
            'config', 'dashboard.html',
        )
    except ImportError:
        pass
    for base in ['/home/alig/41068_ws/src', os.path.expanduser('~/41068_ws/src')]:
        p = os.path.join(base, 'deforestation_monitoring', 'config', 'dashboard.html')
        if os.path.isfile(p):
            return p
    return None


HTML_PATH = _find_dashboard()
PHOTO_DIRS = ['/tmp/deforestation_eval/waste_photos', '/tmp/deforestation_inspections',
              '/tmp/deforestation_eval/evidence_photos']


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/api/data':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            with store_lock:
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
            fresh = time.time() - store.get('husky_yolo_t', 0) < 3.0
            self._serve_cached_camera('husky_yolo' if fresh else 'husky_cam')
        elif path.startswith('/api/photo/'):
            self._serve_photo(os.path.basename(path))
        elif path == '/api/photos':
            self._serve_photo_list()
        else:
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

    def _serve_photo(self, name):
        for folder in PHOTO_DIRS:
            f = os.path.join(folder, name)
            if name.lower().endswith(('.jpg', '.png')) and os.path.isfile(f):
                with open(f, 'rb') as fh:
                    body = fh.read()
                self.send_response(200)
                self.send_header('Content-Type',
                                 'image/png' if name.lower().endswith('.png') else 'image/jpeg')
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                self.wfile.write(body)
                return
        self.send_response(404)
        self.end_headers()

    def _serve_photo_list(self):
        photos = []
        for folder in PHOTO_DIRS:
            try:
                names = os.listdir(folder)
            except OSError:
                continue
            for name in names:
                f = os.path.join(folder, name)
                if not name.lower().endswith(('.jpg', '.png')):
                    continue
                mtime = os.path.getmtime(f)
                if mtime >= store['started']:
                    photos.append({'name': name, 'time': mtime,
                                   'kind': ('waste' if 'waste' in folder else
                                            'evidence' if 'evidence' in folder else 'inspection')})
        photos.sort(key=lambda p: -p['time'])
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(photos[:200]).encode())

    def _serve_cached_camera(self, key):
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
        pass


def start_server():
    if HTML_PATH:
        print(f'[server] Dashboard: {HTML_PATH}')
    else:
        print('[server] WARNING: dashboard.html not found. Only /api/ endpoints available.')
    threading.Thread(target=rosbridge_thread, daemon=True).start()
    server = HTTPServer(('0.0.0.0', 8081), Handler)
    print('[server] Data server on http://0.0.0.0:8081')
    server.serve_forever()


def main(args=None):
    start_server()


if __name__ == '__main__':
    main()
