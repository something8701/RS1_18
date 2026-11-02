#!/usr/bin/env python3
# radio_link.py: The Husky can only report while it has a radio route to the base
import json
import math
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import serialize_message
from rosidl_runtime_py.utilities import get_message
from std_msgs.msg import String

from .radio_model import link_margin

UP, DOWN = 'up', 'down'
LOCAL = '/husky1/local'


def _ids(field):
    return lambda m: tuple(sorted((x.id, x.status) for x in getattr(m, field)))


TOPICS = [
    (DOWN, '/suspicious_areas', 'deforestation_interfaces/msg/SuspiciousArea', False, True, True),
    (DOWN, '/parrot_tree_change_events', 'deforestation_interfaces/msg/TreeChangeEvent', False, True, True),
    (DOWN, '/parrot_tree_baseline_info', 'std_msgs/msg/String', True, False, True),
    (DOWN, '/drone_baseline_status', 'std_msgs/msg/String', False, False,
     lambda m: True if m.data.startswith('frozen') else None),
    (DOWN, '/canopy_change_map', 'nav_msgs/msg/OccupancyGrid', True, False, False),
    (DOWN, '/forest_red_map', 'nav_msgs/msg/OccupancyGrid', True, False, False),
    (DOWN, '/forest_canopy_map', 'nav_msgs/msg/OccupancyGrid', True, False, False),
    (UP, '/waste_sites', 'deforestation_interfaces/msg/WasteSiteArray', True, False, _ids('sites')),
    (UP, '/refined_trees', 'deforestation_interfaces/msg/RefinedTreeArray', True, False, _ids('trees')),
    (UP, '/logging_evidence', 'deforestation_interfaces/msg/EvidenceArray', True, False,
     lambda m: tuple(sorted(x.id for x in m.items))),
    (UP, '/inspection_reports', 'std_msgs/msg/String', False, True, True),
    (UP, '/mission_queue', 'deforestation_interfaces/msg/MissionQueue', False, False,
     lambda m: (m.state, len(m.done), m.current.label)),
    (UP, '/mission_status', 'std_msgs/msg/String', False, False, False),
    (DOWN, '/mission_command', 'std_msgs/msg/String', False, True, True),
    (UP, '/mission_command_ack', 'std_msgs/msg/String', False, True, True),
    (UP, '/mission_control', 'std_msgs/msg/String', True, False, lambda m: m.data),
]


class Channel:

    def __init__(self, direction, topic, keep_all, news):
        self.direction, self.topic, self.keep_all, self.news = direction, topic, keep_all, news
        self.waiting = deque(maxlen=500)
        self.last_key = None
        self.pub = None
        self.delivered = 0
        self.dropped = 0

    def push(self, msg, now):
        if self.news is True:
            is_news = True
        elif self.news:
            is_news = self.news(msg) != self.last_key
        else:
            is_news = False
        if not self.keep_all:
            first = self.waiting[0][1] if self.waiting and self.waiting[0][2] else now
            news = is_news or bool(self.waiting and self.waiting[0][2])
            self.waiting.clear()
            self.waiting.append((msg, first if news else now, news))
        else:
            if len(self.waiting) == self.waiting.maxlen:
                self.dropped += 1
            self.waiting.append((msg, now, is_news))

    def flush(self):
        n = len(self.waiting)
        while self.waiting:
            msg = self.waiting.popleft()[0]
            self.pub.publish(msg)
            if self.news not in (True, False):
                self.last_key = self.news(msg)
        self.delivered += n
        return n

    def oldest_news(self) -> Optional[float]:
        times = [t for _, t, news in self.waiting if news]
        return min(times) if times else None


class RadioLink(Node):

    def __init__(self):
        super().__init__('radio_link')
        p = lambda n, d: self.declare_parameter(n, d).value
        self.base = (float(p('base_x', 0.0)), float(p('base_y', 0.0)), float(p('base_height', 2.0)))
        self.husky_h = float(p('husky_height', 0.8))
        self.drone_alt = float(p('drone_altitude', 10.0))
        self.f_ghz = float(p('frequency_ghz', 2.4))
        self.tx_dbm = float(p('tx_dbm', -6.0))
        self.gain = {'base': float(p('base_gain_dbi', 3.0)), 'drone': float(p('drone_gain_dbi', 2.0)),
                     'husky': float(p('husky_gain_dbi', 0.0))}
        self.sens_dbm = float(p('sensitivity_dbm', -85.0))
        self.fade_db = float(p('fade_margin_db', 10.0))
        self.hyst_db = float(p('hysteresis_db', 1.0))
        self.relay_after = float(p('relay_after', 30.0))
        self.relay_hold = float(p('relay_hold', 5.0))
        self.relay_cooldown = float(p('relay_cooldown', 20.0))
        self.drone_timeout = float(p('drone_timeout', 5.0))

        self.husky = None
        self.drone = None
        self.drone_t = None
        self.no_relay_logged = False
        self.dropped_logged = 0
        self.heights = None
        self.grid_geo = None
        self.up = {'husky-base': False, 'husky-drone': False, 'drone-base': False}
        self.margins: Dict[str, Tuple[float, float, float]] = {}
        self.route = None
        self.contact_since = None
        self.lost_since = None
        self.relay = {'active': False, 'x': 0.0, 'y': 0.0, 'since': 0.0, 'clear_at': None, 'count': 0}
        self.relay_released = -1e9

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        normal = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE)
        self.channels = []
        for direction, topic, type_name, is_latched, keep_all, news in TOPICS:
            ch = Channel(direction, topic, keep_all, news)
            src, dst = (LOCAL + topic, topic) if direction == UP else (topic, LOCAL + topic)
            qos = latched if is_latched else normal
            msg_type = get_message(type_name)
            ch.pub = self.create_publisher(msg_type, dst, qos)
            self.create_subscription(msg_type, src, lambda m, c=ch: c.push(m, self._now()), qos)
            self.channels.append(ch)

        sensor = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Odometry, '/husky1/odometry',
                                 lambda m: setattr(self, 'husky', (m.pose.pose.position.x, m.pose.pose.position.y)), sensor)
        self.create_subscription(Odometry, '/parrot1/odometry', self.drone_cb, sensor)
        self.create_subscription(OccupancyGrid, '/forest_canopy_map', self.canopy_cb, latched)
        self.links_pub = self.create_publisher(String, '/radio/links', 10)
        self.relay_pub = self.create_publisher(String, '/radio/relay_request', 10)
        self.create_timer(0.5, self.tick)
        self.create_timer(1.0, self.publish_links)
        self.get_logger().info(
            f'Radio link ready: base at ({self.base[0]:.1f},{self.base[1]:.1f}), {self.tx_dbm:.0f} dBm, '
            f'sensitivity {self.sens_dbm:.0f} dBm + {self.fade_db:.0f} dB margin, '
            f'{len(self.channels)} gated topics, relay after {self.relay_after:.0f} s of waiting news')

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def drone_cb(self, msg: Odometry):
        self.drone = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.drone_t = self._now()

    def drone_alive(self) -> bool:
        return self.drone_t is not None and self._now() - self.drone_t <= self.drone_timeout

    def canopy_cb(self, msg: OccupancyGrid):
        i = msg.info
        g = np.asarray(msg.data, np.int8).reshape(i.height, i.width).astype(np.float32)
        self.heights = np.where(g >= 0, g / 10.0, -1.0)
        self.grid_geo = (i.resolution, i.origin.position.x, i.origin.position.y)

    def _margin(self, a, b, ga, gb):
        res, ox, oy = self.grid_geo or (1.0, 0.0, 0.0)
        return link_margin(a, b, self.heights, res, ox, oy, self.f_ghz, self.tx_dbm,
                           (ga, gb), self.sens_dbm, self.fade_db)

    def _update_links(self):
        h = (*self.husky, self.husky_h)
        pairs = {'husky-base': (h, self.base, self.gain['husky'], self.gain['base'])}
        if self.drone_alive():
            d = (*self.drone, self.drone_alt)
            pairs['husky-drone'] = (h, d, self.gain['husky'], self.gain['drone'])
            pairs['drone-base'] = (d, self.base, self.gain['drone'], self.gain['base'])
        else:
            for name in ('husky-drone', 'drone-base'):
                self.up[name] = False
                self.margins.pop(name, None)
        for name, (a, b, ga, gb) in pairs.items():
            m = self._margin(a, b, ga, gb)
            self.margins[name] = m
            self.up[name] = m[0] > -self.hyst_db if self.up[name] else m[0] > self.hyst_db
        if self.up['husky-base']:
            route = 'direct'
        elif self.up['husky-drone'] and self.up['drone-base']:
            route = 'relay'
        else:
            route = None
        if route != self.route:
            name = 'husky-base'
            m, dist, fol = self.margins[name]
            if route is None:
                self.lost_since = self._now()
                self.get_logger().warn(
                    f'Husky out of radio contact at ({self.husky[0]:.1f},{self.husky[1]:.1f}): '
                    f'{dist:.0f} m to the base, {fol:.0f} m of it through trees ({m:+.1f} dB); '
                    f'its messages wait')
            else:
                self.no_relay_logged = False
                self.get_logger().info(
                    f'Husky in radio contact ({"through the drone" if route == "relay" else "direct"}, '
                    f'base link {m:+.1f} dB)')
                self.contact_since = self._now()
            self.route = route

    def _best_relay_point(self):
        best = None
        for k in np.linspace(0.3, 1.0, 15):
            x = self.base[0] + (self.husky[0] - self.base[0]) * k
            y = self.base[1] + (self.husky[1] - self.base[1]) * k
            d = (x, y, self.drone_alt)
            m1 = self._margin((*self.husky, self.husky_h), d, self.gain['husky'], self.gain['drone'])[0]
            m2 = self._margin(d, self.base, self.gain['drone'], self.gain['base'])[0]
            if best is None or min(m1, m2) > best[0]:
                best = (min(m1, m2), x, y)
        return best

    def tick(self):
        if self.husky is None:
            return
        self._update_links()
        now = self._now()
        if self.route is not None:
            for ch in self.channels:
                ch.flush()
        oldest = [t for t in (ch.oldest_news() for ch in self.channels) if t is not None]
        waiting_news = min(oldest) if oldest else None
        dropped = sum(ch.dropped for ch in self.channels)
        if dropped > self.dropped_logged:
            self.get_logger().warn(
                f'{dropped - self.dropped_logged} waiting messages dropped (queue full): '
                + ', '.join(f'{ch.topic} {ch.dropped}' for ch in self.channels if ch.dropped))
            self.dropped_logged = dropped

        r = self.relay
        if not r['active']:
            if (self.route is None and waiting_news is not None and now - waiting_news >= self.relay_after
                    and now - self.relay_released >= self.relay_cooldown and self.drone_alive()):
                score, x, y = self._best_relay_point()
                if score <= self.hyst_db:
                    if not self.no_relay_logged:
                        self.get_logger().warn(
                            f'No relay point closes both links (best {score:+.1f} dB): '
                            f'the messages wait for the Husky to come back')
                        self.no_relay_logged = True
                else:
                    r.update(active=True, x=x, y=y, since=now, clear_at=None, count=r['count'] + 1)
                    waiting = sum(len(ch.waiting) for ch in self.channels)
                    self.get_logger().warn(
                        f'Calling the drone to relay: {waiting} messages waiting for '
                        f'{now - waiting_news:.0f} s; hover at ({x:.1f},{y:.1f}) (weaker link {score:+.1f} dB)')
        elif not self.drone_alive():
            self.get_logger().warn('The drone went quiet during a relay: relay cancelled')
            r['active'] = False
            self.relay_released = now
        else:
            _, r['x'], r['y'] = self._best_relay_point()
            if self.route is None or waiting_news is not None:
                r['clear_at'] = None
            elif r['clear_at'] is None:
                r['clear_at'] = now
            if r['clear_at'] is not None and now - r['clear_at'] >= self.relay_hold:
                self.get_logger().info(
                    f'Relay done after {now - r["since"]:.0f} s: everything delivered, the drone resumes its survey')
                r['active'] = False
                self.relay_released = now
            elif self.route == 'direct':
                r['active'] = False
                self.relay_released = now
        self.relay_pub.publish(String(data=json.dumps(
            {'active': r['active'], 'x': round(r['x'], 2), 'y': round(r['y'], 2), 'count': r['count']})))

    def publish_links(self):
        if self.husky is None:
            return
        now = self._now()
        oldest = [t for t in (ch.oldest_news() for ch in self.channels) if t is not None]
        info = {
            'base': [round(v, 2) for v in self.base[:2]],
            'husky': [round(v, 2) for v in self.husky],
            'drone': [round(v, 2) for v in self.drone] if self.drone_alive() else None,
            'links': {k: {'up': self.up[k], 'margin': round(m, 1), 'distance': round(d, 1),
                          'foliage': round(f, 1)} for k, (m, d, f) in self.margins.items()},
            'route': self.route,
            'waiting': {UP: sum(len(c.waiting) for c in self.channels if c.direction == UP),
                        DOWN: sum(len(c.waiting) for c in self.channels if c.direction == DOWN)},
            'news_wait_s': round(now - min(oldest), 1) if oldest else 0.0,
            'delivered': {UP: sum(c.delivered for c in self.channels if c.direction == UP),
                          DOWN: sum(c.delivered for c in self.channels if c.direction == DOWN)},
            'dropped': sum(c.dropped for c in self.channels),
            'out_of_contact_s': round(now - self.lost_since, 1) if self.route is None and self.lost_since else 0.0,
            'relay': {'active': self.relay['active'], 'x': round(self.relay['x'], 2),
                      'y': round(self.relay['y'], 2), 'count': self.relay['count']},
        }
        self.links_pub.publish(String(data=json.dumps(info)))


def main(args=None):
    rclpy.init(args=args)
    node = RadioLink()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
