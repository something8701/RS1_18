#!/usr/bin/env python3
# drone_prior_map.py: The drone's tree map as a cost mask for the Husky's planner
import json
import math
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import rclpy
from deforestation_interfaces.msg import TreeChangeEvent
from nav2_msgs.msg import CostmapFilterInfo
from nav_msgs.msg import OccupancyGrid
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

Tree = Tuple[float, float, str]


def prior_mask(trees: Iterable[Tree], res: float, ox: float, oy: float,
               width: int, height: int, core: Dict[str, float], fade: float,
               cost: float) -> np.ndarray:
    mask = np.zeros((height, width), np.float32)
    for x, y, species in trees:
        c = core[species]
        reach = c + fade
        ix0 = max(0, int(math.floor((x - reach - ox) / res)))
        ix1 = min(width, int(math.ceil((x + reach - ox) / res)) + 1)
        iy0 = max(0, int(math.floor((y - reach - oy) / res)))
        iy1 = min(height, int(math.ceil((y + reach - oy) / res)) + 1)
        if ix0 >= ix1 or iy0 >= iy1:
            continue
        cx = ox + (np.arange(ix0, ix1) + 0.5) * res
        cy = oy + (np.arange(iy0, iy1) + 0.5) * res
        d = np.hypot(cx[None, :] - x, cy[:, None] - y)
        v = cost * np.clip((reach - d) / max(fade, 1e-6), 0.0, 1.0)
        v[d <= c] = cost
        np.maximum(mask[iy0:iy1, ix0:ix1], v, out=mask[iy0:iy1, ix0:ix1])
    return np.rint(mask).astype(np.int8)


def canopy_top(grid: Optional[OccupancyGrid], x: float, y: float,
               radius: float = 0.75) -> Optional[float]:
    if grid is None:
        return None
    i = grid.info
    g = np.asarray(grid.data, np.int8).reshape(i.height, i.width)
    k = int(math.ceil(radius / i.resolution))
    ix = int((x - i.origin.position.x) / i.resolution)
    iy = int((y - i.origin.position.y) / i.resolution)
    sub = g[max(0, iy - k):iy + k + 1, max(0, ix - k):ix + k + 1]
    sub = sub[sub >= 0]
    return float(sub.max()) / 10.0 if sub.size else None


def crown_gone(change: Optional[OccupancyGrid], x: float, y: float,
               radius: float = 1.0) -> bool:
    if change is None:
        return False
    i = change.info
    g = np.asarray(change.data, np.int8).reshape(i.height, i.width)
    k = int(math.ceil(radius / i.resolution))
    ix = int((x - i.origin.position.x) / i.resolution)
    iy = int((y - i.origin.position.y) / i.resolution)
    sub = g[max(0, iy - k):iy + k + 1, max(0, ix - k):ix + k + 1].astype(int)
    seen = sub[sub != -1]
    return bool(seen.size) and np.count_nonzero(seen <= -40) * 2 >= seen.size


def classify(species: Optional[str], top: Optional[float], pine_below: float) -> str:
    if species in ('pine', 'oak'):
        return species
    if top is not None and top >= pine_below:
        return 'oak'
    return 'pine'


class DronePriorMap(Node):

    def __init__(self):
        super().__init__('drone_prior_map')
        p = lambda n, d: self.declare_parameter(n, d).value
        self.frame = str(p('frame', 'husky1_odom'))
        self.res = float(p('resolution', 0.25))
        self.size = float(p('size', 80.0))
        self.cost = float(p('cost', 85.0))
        self.core = {'pine': float(p('pine_core', 2.2)), 'oak': float(p('oak_core', 2.1))}
        self.fade = float(p('fade', 0.8))
        self.pine_below = float(p('pine_below', 5.4))
        self.mask_topic = str(p('mask_topic', '/husky1/drone_prior_mask'))

        self.trees: Dict[int, Tuple[float, float, Optional[str]]] = {}
        self.cut = set()
        self.canopy = None
        self.change = None
        self.gone = set()
        self.dirty = True

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.mask_pub = self.create_publisher(OccupancyGrid, self.mask_topic, latched)
        self.info_pub = self.create_publisher(
            CostmapFilterInfo, str(p('info_topic', '/husky1/drone_prior_info')), latched)
        self.create_subscription(String, '/parrot_tree_baseline_info', self.baseline_cb, latched)
        self.create_subscription(TreeChangeEvent, '/parrot_tree_change_events', self.event_cb, 50)
        self.create_subscription(OccupancyGrid, '/canopy_change_map', self.change_cb, latched)
        for durability in (DurabilityPolicy.TRANSIENT_LOCAL, DurabilityPolicy.VOLATILE):
            self.create_subscription(
                OccupancyGrid, '/forest_canopy_map', lambda m: setattr(self, 'canopy', m),
                QoSProfile(depth=1, durability=durability, reliability=ReliabilityPolicy.RELIABLE))
        self.create_timer(1.0, self.publish_if_dirty)
        self.get_logger().info(
            f'Drone prior map ready: pine {self.core["pine"]} m, oak {self.core["oak"]} m '
            f'at cost {self.cost:.0f}, fading over {self.fade} m, mask {self.mask_topic}')

    def baseline_cb(self, msg: String):
        try:
            info = json.loads(msg.data)
        except ValueError:
            return
        trees = {int(t['id']): (float(t['x']), float(t['y']), t.get('species'))
                 for t in info.get('trees', [])}
        if {k: v[:2] for k, v in trees.items()} != {k: v[:2] for k, v in self.trees.items()}:
            self.cut.clear()
            self.gone.clear()
        self.trees = trees
        self.dirty = True

    def event_cb(self, msg: TreeChangeEvent):
        if msg.event_type == 'LOST' and msg.tree_id not in self.cut:
            self.cut.add(int(msg.tree_id))
            self.dirty = True

    def change_cb(self, msg: OccupancyGrid):
        self.change = msg
        gone = {tid for tid, (x, y, _) in self.trees.items() if crown_gone(msg, x, y)}
        if gone != self.gone:
            self.gone = gone
            self.dirty = True

    def publish_if_dirty(self):
        if not self.dirty:
            return
        self.dirty = False
        trees, by_height = [], 0
        for tid, (x, y, species) in self.trees.items():
            if tid in self.cut or tid in self.gone:
                continue
            kind = classify(species, canopy_top(self.canopy, x, y), self.pine_below)
            by_height += species not in ('pine', 'oak')
            trees.append((x, y, kind))
        n = int(round(self.size / self.res))
        o = -self.size / 2.0
        mask = prior_mask(trees, self.res, o, o, n, n, self.core, self.fade, self.cost)

        grid = OccupancyGrid()
        grid.header.frame_id = self.frame
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.info.resolution = self.res
        grid.info.width = grid.info.height = n
        grid.info.origin.position.x = grid.info.origin.position.y = o
        grid.info.origin.orientation.w = 1.0
        grid.data = mask.ravel().tolist()
        self.mask_pub.publish(grid)
        self.info_pub.publish(CostmapFilterInfo(
            header=grid.header, type=0, filter_mask_topic=self.mask_topic,
            base=0.0, multiplier=1.0))
        pines = sum(t[2] == 'pine' for t in trees)
        self.get_logger().info(
            f'Drone prior: {len(trees)} trees ({pines} pine, {len(trees) - pines} oak, '
            f'{by_height} by height), {len(self.cut | self.gone)} cut left out; '
            f'{100.0 * np.mean(mask >= 70):.0f}% of the map closed (>= 70)')


def main(args=None):
    rclpy.init(args=args)
    node = DronePriorMap()
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
