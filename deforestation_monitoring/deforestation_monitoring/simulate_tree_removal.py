#!/usr/bin/env python3
"""
Tree Removal Simulator: deletes a known cluster of trees in Gazebo so the
perception pipeline can detect a real canopy/tree loss.

It waits for the mapping baselines, then deletes each target tree through the
Gazebo entity-remove service. The `ign service` CLI is used because it talks to
the Ignition transport directly and needs no ROS service bridge. The removed
trees' centroid is published on /ground_truth_events so evaluate_mission can
score the detection pipeline against a known clearing.

Tree positions are read from the installed world SDF (any world), with
hardcoded simple_trees / dense_forest tables as a fallback. Removal waits
for the parrot_tree_tracker "frozen: N trees" baseline, not just any
/drone_baseline_status message.
"""

import os
import subprocess
import tempfile
from typing import Dict, List, Tuple

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import Header, String
from geometry_msgs.msg import Point
from std_srvs.srv import Trigger
from deforestation_interfaces.msg import SuspiciousArea

from .tree_detection import parse_tree_truth


# Ground-truth (x, y) positions of the 13 trees in simple_trees.sdf.
SIMPLE_TREES: Dict[str, Tuple[float, float]] = {
    'oak_1': (-23.9, -28.8),
    'oak_2': (-27.9, -15.2),
    'oak_3': (-24.3, -2.7),
    'pine_4': (-25.0, 10.2),
    'oak_5': (-14.2, -24.3),
    'pine_6': (-9.9, -14.7),
    'oak_7': (-14.8, 4.7),
    'pine_8': (-13.0, 12.9),
    'pine_9': (1.3, -8.2),
    'oak_10': (3.6, 1.9),
    'pine_11': (2.6, 10.4),
    'pine_12': (17.9, -22.2),
    'pine_13': (12.2, -10.9),
}

# Ground-truth (x, y) positions of the 214 trees in dense_forest.sdf.
DENSE_FOREST_TREES: Dict[str, Tuple[float, float]] = {
    'oak_1': (-36.9, -35.9), 'oak_2': (-33.4, -31.7), 'pine_3': (-36.1, -24.6),
    'oak_4': (-34.6, -18.8), 'pine_5': (-35.6, -16.4), 'pine_6': (-35.5, -10.6),
    'pine_7': (-34.1, -4.9), 'pine_8': (-35.8, 0.5), 'pine_9': (-34.2, 3.2),
    'pine_10': (-33.6, 11.5), 'pine_11': (-35.5, 14.5), 'oak_12': (-34.3, 18.3),
    'oak_13': (-35.5, 27.0), 'oak_14': (-33.6, 31.1), 'pine_15': (-35.4, 33.3),
    'oak_16': (-30.0, -33.5), 'pine_17': (-31.0, -29.8), 'pine_18': (-30.4, -26.1),
    'pine_19': (-31.8, -21.6), 'pine_20': (-30.5, -14.6), 'pine_21': (-28.6, -12.0),
    'oak_22': (-30.6, -5.8), 'oak_23': (-28.5, -0.9), 'oak_24': (-31.6, 5.5),
    'oak_25': (-28.9, 10.2), 'pine_26': (-30.0, 13.4), 'pine_27': (-31.0, 20.3),
    'pine_28': (-28.1, 25.1), 'oak_29': (-31.3, 30.1), 'pine_30': (-29.0, 35.8),
    'oak_31': (-24.4, -35.2), 'oak_32': (-26.7, -31.9), 'pine_33': (-26.8, -21.7),
    'oak_34': (-26.0, -14.3), 'pine_35': (-23.5, -9.7), 'pine_36': (-26.2, -6.6),
    'oak_37': (-23.5, -1.8), 'oak_38': (-24.1, 6.2), 'oak_39': (-25.2, 9.7),
    'oak_40': (-26.7, 16.2), 'oak_41': (-23.7, 26.9), 'oak_42': (-25.1, 28.9),
    'pine_43': (-27.0, 34.6), 'pine_44': (-19.2, -34.1), 'oak_45': (-20.8, -28.1),
    'oak_46': (-21.8, -24.7), 'oak_47': (-18.6, -16.7), 'oak_48': (-18.2, -9.7),
    'pine_49': (-20.3, -4.7), 'pine_50': (-19.1, -1.0), 'oak_51': (-18.3, 6.7),
    'oak_52': (-21.1, 9.1), 'pine_53': (-20.5, 13.6), 'pine_54': (-18.1, 20.6),
    'oak_55': (-21.6, 26.5), 'pine_56': (-21.4, 29.1), 'oak_57': (-19.5, 34.1),
    'oak_58': (-15.3, -35.9), 'pine_59': (-16.0, -29.0), 'oak_60': (-16.6, -23.2),
    'oak_61': (-13.7, -19.7), 'pine_62': (-13.3, -8.1), 'pine_63': (-16.0, -6.6),
    'oak_64': (-14.0, 1.7), 'pine_65': (-13.8, 6.5), 'oak_66': (-15.4, 10.7),
    'pine_67': (-13.9, 13.4), 'oak_68': (-13.7, 19.8), 'oak_69': (-14.4, 24.6),
    'pine_70': (-15.9, 30.6), 'pine_71': (-16.5, 36.9), 'oak_72': (-10.6, -28.9),
    'oak_73': (-9.7, -26.0), 'oak_74': (-8.8, -19.3), 'pine_75': (-11.7, -14.3),
    'pine_76': (-11.5, -8.4), 'pine_77': (-9.2, -4.8), 'oak_78': (-8.4, 0.5),
    'pine_79': (-11.2, 6.1), 'oak_80': (-10.2, 8.9), 'oak_81': (-10.9, 15.7),
    'oak_82': (-9.0, 20.7), 'oak_83': (-11.8, 26.9), 'pine_84': (-8.5, 30.3),
    'oak_85': (-11.0, 35.6), 'pine_86': (-6.1, -33.8), 'pine_87': (-3.6, -25.2),
    'pine_88': (-3.4, -19.5), 'pine_89': (-4.8, -13.4), 'pine_90': (-4.0, -8.6),
    'pine_91': (-5.9, -6.7), 'pine_92': (-3.4, -1.7), 'pine_93': (-4.2, 3.3),
    'oak_94': (-6.2, 9.7), 'pine_95': (-3.9, 14.5), 'oak_96': (-5.4, 21.6),
    'pine_97': (-5.9, 24.9), 'oak_98': (-3.6, 31.7), 'pine_99': (-4.6, 35.7),
    'oak_100': (-0.2, -36.8), 'pine_101': (-0.6, -28.5), 'pine_102': (-1.7, -26.9),
    'oak_103': (-1.7, -18.2), 'oak_104': (-1.2, -16.9), 'pine_105': (-1.5, -8.2),
    'oak_106': (-1.3, -4.6), 'pine_107': (-1.3, -0.8), 'oak_108': (1.6, 4.5),
    'pine_109': (0.8, 10.5), 'oak_110': (0.4, 13.5), 'pine_111': (-0.3, 19.5),
    'pine_112': (-0.0, 29.7), 'pine_113': (-0.3, 35.9), 'pine_114': (6.7, -33.9),
    'pine_115': (4.0, -28.3), 'oak_116': (4.9, -24.6), 'pine_117': (4.3, -20.0),
    'oak_118': (6.5, -14.2), 'oak_119': (3.8, -11.0), 'oak_120': (5.6, -5.0),
    'pine_121': (4.6, -1.0), 'pine_122': (5.1, 4.7), 'oak_123': (4.8, 9.2),
    'oak_124': (6.0, 16.8), 'pine_125': (4.9, 20.9), 'pine_126': (6.3, 24.2),
    'oak_127': (4.1, 36.9), 'pine_128': (11.0, -36.6), 'pine_129': (9.8, -31.3),
    'oak_130': (8.3, -25.0), 'oak_131': (11.2, -18.2), 'oak_132': (11.1, -14.6),
    'pine_133': (9.5, -8.4), 'oak_134': (10.5, -4.6), 'oak_135': (9.1, -1.7),
    'oak_136': (9.6, 5.8), 'oak_137': (10.8, 10.8), 'oak_138': (11.2, 15.5),
    'oak_139': (10.2, 20.6), 'pine_140': (8.7, 24.2), 'pine_141': (9.9, 29.2),
    'pine_142': (10.0, 33.3), 'oak_143': (15.4, -36.9), 'oak_144': (15.7, -30.9),
    'pine_145': (15.6, -23.3), 'pine_146': (15.0, -21.6), 'pine_147': (13.4, -16.4),
    'pine_148': (16.8, -8.7), 'oak_149': (14.0, -6.5), 'pine_150': (15.2, -0.3),
    'pine_151': (15.0, 4.8), 'oak_152': (16.7, 9.8), 'oak_153': (16.2, 20.8),
    'pine_154': (13.4, 27.0), 'pine_155': (15.6, 28.6), 'oak_156': (14.1, 36.3),
    'oak_157': (18.6, -36.2), 'pine_158': (19.1, -28.8), 'pine_159': (19.1, -25.8),
    'pine_160': (19.8, -20.1), 'pine_161': (19.3, -16.2), 'pine_162': (21.1, -11.8),
    'pine_163': (20.7, -3.7), 'oak_164': (18.5, 1.9), 'oak_165': (19.8, 5.1),
    'pine_166': (20.6, 8.6), 'pine_167': (20.8, 15.6), 'oak_168': (20.5, 21.5),
    'oak_169': (18.9, 25.7), 'oak_170': (19.7, 28.4), 'oak_171': (20.8, 36.6),
    'oak_172': (24.5, -36.4), 'oak_173': (23.3, -28.5), 'pine_174': (25.3, -26.1),
    'oak_175': (23.0, -18.8), 'pine_176': (26.8, -13.0), 'pine_177': (24.8, -10.8),
    'pine_178': (24.5, -5.6), 'oak_179': (24.6, 3.0), 'oak_180': (26.0, 11.3),
    'oak_181': (24.4, 15.5), 'pine_182': (26.6, 20.6), 'pine_183': (23.5, 31.6),
    'pine_184': (23.5, 35.2), 'pine_185': (29.0, -33.7), 'oak_186': (29.4, -31.5),
    'pine_187': (28.9, -24.3), 'oak_188': (31.2, -20.5), 'pine_189': (28.5, -15.5),
    'oak_190': (31.3, -8.2), 'oak_191': (30.5, -3.2), 'pine_192': (30.5, -0.1),
    'oak_193': (28.2, 3.1), 'pine_194': (31.5, 8.5), 'pine_195': (30.9, 16.5),
    'oak_196': (31.8, 19.1), 'pine_197': (31.8, 26.4), 'oak_198': (31.9, 29.4),
    'oak_199': (31.8, 36.3), 'oak_200': (35.4, -36.9), 'oak_201': (36.1, -31.2),
    'oak_202': (36.3, -26.1), 'oak_203': (36.9, -18.0), 'oak_204': (36.5, -16.5),
    'pine_205': (33.1, -10.9), 'oak_206': (36.0, -5.1), 'pine_207': (33.9, 0.5),
    'pine_208': (36.9, 3.2), 'oak_209': (33.4, 10.8), 'pine_210': (33.6, 13.5),
    'pine_211': (35.4, 19.2), 'oak_212': (35.8, 26.4), 'pine_213': (36.6, 29.7),
    'pine_214': (33.4, 34.7),
}


def positions_for_world(world: str) -> Dict[str, Tuple[float, float]]:
    """Return the name->(x, y) map for the given Gazebo world.

    Read from the installed world SDF, so every world (sparse_trees,
    cluster_test, ...) gets its real tree positions. The hardcoded tables
    are only a fallback when the SDF cannot be found.
    """
    try:
        from ament_index_python.packages import get_package_share_directory
        path = os.path.join(
            get_package_share_directory('41068_ignition_bringup'),
            'worlds', f'{world}.sdf')
        if os.path.isfile(path):
            truth = parse_tree_truth(path)
            if truth:
                return {name: (x, y) for name, x, y in truth}
    except Exception:  # noqa: BLE001 - any lookup failure -> fallback
        pass
    if world == 'dense_forest':
        return DENSE_FOREST_TREES
    return SIMPLE_TREES


def remove_model(world: str, name: str) -> Tuple[bool, str]:
    """Delete a model from the running Ignition world. Returns (ok, output)."""
    req = f'name: "{name}" type: MODEL'
    cmd = [
        'ign', 'service', '-s', f'/world/{world}/remove',
        '--timeout', '2000',
        '--reqtype', 'ignition.msgs.Entity',
        '--reptype', 'ignition.msgs.Boolean',
        '--req', req,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15.0)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return False, str(exc)
    out = (r.stdout + r.stderr).strip()
    ok = r.returncode == 0 and 'error' not in out.lower() \
        and 'data: false' not in out.lower()
    return ok, out


class TreeRemoval(Node):
    """Deletes trees in Gazebo and publishes the clearing as ground truth."""

    def __init__(self):
        super().__init__('simulate_tree_removal')

        self.declare_parameter('world', 'simple_trees',
            descriptor=ParameterDescriptor(description='Gazebo world name'))
        self.declare_parameter('tree_names',
            ['pine_9', 'oak_10', 'pine_11', 'pine_13'],
            descriptor=ParameterDescriptor(
                description='Explicit tree model names to remove (used when '
                            'use_center_radius is false)'))
        self.declare_parameter('use_center_radius', False,
            descriptor=ParameterDescriptor(
                description='Select trees within radius of centre from '
                            'TREE_POSITIONS instead of using tree_names'))
        self.declare_parameter('center', [-22.95, 27.85],
            descriptor=ParameterDescriptor(description='Clearing centre (x, y)'))
        self.declare_parameter('radius', 3.0,
            descriptor=ParameterDescriptor(
                description='Clearing radius (m) used with centre'))
        self.declare_parameter('frame_id', 'husky1_map',
            descriptor=ParameterDescriptor(description='Ground-truth frame'))
        self.declare_parameter('require_both_baselines', True,
            descriptor=ParameterDescriptor(
                description='Wait for both tree_mapper and scan_mapper '
                            'baselines before removing trees'))
        self.declare_parameter('trigger_delay', 5.0,
            descriptor=ParameterDescriptor(
                description='Seconds after baselines before auto-triggering'))
        self.declare_parameter('disturbance_type', 'cut',
            descriptor=ParameterDescriptor(
                description="Disturbance evidence to leave behind: 'cut' spawns a "
                            "short stump cylinder, 'windthrow' spawns a fallen trunk. "
                            'Gives the classifier real evidence to distinguish CUT from '
                            'NATURAL instead of deleting the whole model.'))

        self.world = self.get_parameter('world').value
        self.tree_names = self.get_parameter('tree_names').value
        self.use_center = self.get_parameter('use_center_radius').value
        self.center = self.get_parameter('center').value
        self.radius = self.get_parameter('radius').value
        self.frame_id = self.get_parameter('frame_id').value
        self.require_both = self.get_parameter('require_both_baselines').value
        self.trigger_delay = self.get_parameter('trigger_delay').value
        self.disturbance_type = self.get_parameter('disturbance_type').value

        self.target_trees: List[str] = self._select_trees()
        known = positions_for_world(self.world)
        missing = [n for n in self.target_trees if n not in known]
        if missing:
            self.get_logger().warn(
                f'Target trees not in world {self.world}: {missing}. '
                f'They cannot be removed or scored.')
        self.tree_baseline_ready = False
        self.drone_baseline_ready = False
        self._triggered = False
        self._delay_until = None

        alert_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(
            String, '/baseline_status', self._tree_baseline_cb, alert_qos)
        self.create_subscription(
            String, '/drone_baseline_status', self._drone_baseline_cb, alert_qos)
        self.truth_pub = self.create_publisher(
            SuspiciousArea, '/ground_truth_events', alert_qos)
        self.trigger_srv = self.create_service(
            Trigger, '~/trigger', self._manual_trigger)

        self.create_timer(2.0, self._poll)
        self.get_logger().info(
            f'Tree Removal ready. World={self.world}, targets={self.target_trees}. '
            f'Waiting for baselines (both={self.require_both}).')

    # ── Tree selection ───────────────────────────────────────────────

    def _select_trees(self) -> List[str]:
        if not self.use_center:
            return list(self.tree_names)
        cx, cy = self.center[0], self.center[1]
        chosen = []
        for name, (tx, ty) in positions_for_world(self.world).items():
            if (tx - cx) ** 2 + (ty - cy) ** 2 <= self.radius ** 2:
                chosen.append(name)
        if not chosen:
            self.get_logger().warn(
                f'No trees within radius {self.radius} of ({cx},{cy}); '
                f'falling back to tree_names={self.tree_names}')
            return list(self.tree_names)
        return chosen

    # ── Baseline handling ────────────────────────────────────────────

    def _tree_baseline_cb(self, _msg):
        if not self.tree_baseline_ready:
            self.tree_baseline_ready = True
            self.get_logger().info('tree_mapper baseline received')

    def _drone_baseline_cb(self, msg):
        # /drone_baseline_status is shared: parrot_tree_tracker publishes
        # "waiting: coverage ..." until its baseline exists. Only "frozen: N
        # trees" means the trees are baselined; cutting earlier would make the
        # loss invisible.
        if self.drone_baseline_ready or not msg.data.startswith('frozen'):
            return
        self.drone_baseline_ready = True
        self.get_logger().info(f'parrot tree baseline received: {msg.data}')

    def _ready(self) -> bool:
        if self.require_both:
            return self.tree_baseline_ready and self.drone_baseline_ready
        return self.tree_baseline_ready or self.drone_baseline_ready

    # ── Triggering ───────────────────────────────────────────────────

    def _manual_trigger(self, _request, response):
        removed = self._remove_targets()
        if removed:
            response.success = True
            response.message = f'Removed {len(removed)} trees: {removed}'
        else:
            response.success = False
            response.message = 'No trees removed (see log)'
        return response

    def _poll(self):
        if self._triggered or not self._ready():
            return
        now = self.get_clock().now().nanoseconds / 1e9
        if self._delay_until is None:
            self._delay_until = now + self.trigger_delay
            self.get_logger().info(
                f'Baselines ready; removing trees in {self.trigger_delay:.0f}s')
            return
        if now < self._delay_until:
            return
        self._triggered = True
        removed = self._remove_targets()
        if removed:
            self._publish_ground_truth(removed)

    # ── Gazebo deletion ──────────────────────────────────────────────

    def _remove_tree(self, name: str) -> bool:
        ok, out = remove_model(self.world, name)
        if not ok:
            self.get_logger().error(f'remove {name} failed: {out}')
        return ok

    def _spawn_replacement(self, name: str, x: float, y: float) -> bool:
        """Spawn disturbance evidence where the tree used to be.

        'cut' -> short vertical stump cylinder (deliberate removal).
        'windthrow' -> fallen horizontal trunk (natural fall).
        Gives the ground inspection real geometry to classify instead of an
        empty hole.
        """
        if self.disturbance_type == 'windthrow':
            geom = '<cylinder><radius>0.12</radius><length>3.0</length></cylinder>'
            rot = '0 1.5708 0'
            mat = ('<material><ambient>0.4 0.25 0.15 1</ambient>'
                   '<diffuse>0.4 0.25 0.15 1</diffuse></material>')
            mname = f'windthrow_{name}'
        else:  # cut
            geom = '<cylinder><radius>0.18</radius><length>0.5</length></cylinder>'
            rot = '0 0 0'
            mat = ('<material><ambient>0.35 0.2 0.1 1</ambient>'
                   '<diffuse>0.35 0.2 0.1 1</diffuse></material>')
            mname = f'stump_{name}'

        sdf = (f'<sdf version="1.6"><model name="{mname}"><static>true</static>'
               f'<pose>{x} {y} 0 {rot}</pose>'
               f'<link name="link"><visual name="v"><geometry>{geom}</geometry>{mat}</visual>'
               f'<collision name="c"><geometry>{geom}</geometry></collision></link>'
               f'</model></sdf>')

        fd, path = tempfile.mkstemp(suffix='.sdf', prefix='disturbance_', text=True)
        with os.fdopen(fd, 'w') as fh:
            fh.write(sdf)

        req = f'sdf_filename: "{path}" name: "{mname}"'
        cmd = [
            'ign', 'service', '-s', f'/world/{self.world}/create',
            '--timeout', '2000',
            '--reqtype', 'ignition.msgs.EntityFactory',
            '--reptype', 'ignition.msgs.Boolean',
            '--req', req,
        ]
        try:
            r = subprocess.run(
                cmd, capture_output=True, text=True, timeout=15.0)
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            self.get_logger().error(f'spawn {mname} failed: {exc}')
            return False
        ok = r.returncode == 0 and 'error' not in (r.stdout + r.stderr).lower()
        if ok:
            self.get_logger().info(f'spawned {mname} at ({x:.1f},{y:.1f})')
        else:
            self.get_logger().error(
                f'spawn {mname} returned {r.returncode}: '
                f'{(r.stdout + r.stderr).strip()}')
        return ok

    def _remove_targets(self) -> List[str]:
        removed = []
        for name in self.target_trees:
            if not self._remove_tree(name):
                continue
            removed.append(name)
            self.get_logger().info(f'removed tree {name}')
            pos = positions_for_world(self.world).get(name)
            if pos is not None:
                self._spawn_replacement(name, pos[0], pos[1])
        return removed

    # ── Ground truth ─────────────────────────────────────────────────

    def _publish_ground_truth(self, removed: List[str]):
        tree_map = positions_for_world(self.world)
        positions = [tree_map[n] for n in removed if n in tree_map]
        if not positions:
            self.get_logger().warn('No known positions for removed trees')
            return
        cx = sum(p[0] for p in positions) / len(positions)
        cy = sum(p[1] for p in positions) / len(positions)
        area = float(len(removed) * 12.0)  # rough per-tree footprint estimate

        flag = SuspiciousArea()
        flag.header = Header(
            stamp=self.get_clock().now().to_msg(), frame_id=self.frame_id)
        flag.position = Point(x=float(cx), y=float(cy), z=0.0)
        flag.type = 'CLEARING'
        flag.confidence = 100.0
        flag.area_m2 = area
        flag.severity = 90.0
        flag.description = (
            f'Simulated clearing: removed {len(removed)} trees '
            f'({", ".join(removed)})')
        self.truth_pub.publish(flag)
        self.get_logger().info(
            f'GROUND TRUTH published at ({cx:.1f},{cy:.1f}) '
            f'for {len(removed)} trees: {removed}')


def main(args=None):
    rclpy.init(args=args)
    node = TreeRemoval()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
