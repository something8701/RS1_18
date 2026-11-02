# test_logging_evidence.py: Tests for logging_evidence (stump/log models, placement, scoring) and the pure helpers of …
import math
import random

import numpy as np

from deforestation_monitoring.logging_evidence import (
    footprint_radius, log_sdf, log_spot_near, model_z, params_for, random_params,
    score_evidence, stump_sdf, surface_points)
from deforestation_monitoring.yolo_detector import (
    EvidenceTracker, box_distance, pixel_ray)


def test_params_repeatable_and_mission_stumps_below_lidar():
    assert params_for('stump', 'oak_3') == params_for('stump', 'oak_3')
    assert params_for('log', 'oak_3') != params_for('log', 'pine_9')
    for k in range(50):
        assert params_for('stump', f'pine_{k}')['h'] <= 0.7


def test_sdf_builds_and_collision_flag():
    p = random_params('stump', random.Random(1))
    s = stump_sdf('st', 1.0, 2.0, 0.3, p)
    assert '<collision' in s and 'name="cut"' in s
    assert '<collision' not in stump_sdf('st', 1.0, 2.0, 0.3, p, collision=False)
    lp = random_params('log', random.Random(2))
    assert 'name="end_a"' in log_sdf('lg', 0, 0, 0, lp)


def test_surface_points_match_size_and_pose():
    p = {'kind': 'stump', 'r': 0.2, 'h': 0.5, 'roots': 4, 'root_yaw': 0.0}
    pts = surface_points(p, 5.0, -3.0, 0.0)
    d = np.hypot(pts[:, 0] - 5.0, pts[:, 1] + 3.0)
    assert d.max() <= footprint_radius(p) + 1e-6
    assert abs(pts[:, 2].max() - 0.516) < 1e-6 and pts[:, 2].min() >= 0.0
    lp = {'kind': 'log', 'r': 0.15, 'length': 3.0, 'stubs': 0, 'stub_seed': 0}
    lpts = surface_points(lp, 0.0, 0.0, math.pi / 2)
    assert abs(np.ptp(lpts[:, 1]) - 3.012) < 1e-3 and np.ptp(lpts[:, 0]) <= 0.31
    assert abs(lpts[:, 2].min()) < 1e-6 and abs(lpts[:, 2].max() - 0.3) < 1e-6
    assert model_z(lp) == 0.15


def test_log_spot_clear_of_trunks_and_bag():
    trees = [('cut', 0.0, 0.0), ('a', 3.0, 0.0), ('b', -3.0, 0.5), ('c', 0.0, 3.5)]
    spot = log_spot_near(trees[0], trees, 3.0, avoid=[(0.0, -3.0)])
    assert spot is not None
    x, y, yaw = spot
    for s in np.linspace(-1.5, 1.5, 9):
        px, py = x + s * math.cos(yaw), y + s * math.sin(yaw)
        assert all(math.hypot(px - tx, py - ty) > 0.9 for n, tx, ty in trees if n != 'cut')
        assert math.hypot(px, py + 3.0) > 1.5


def test_score_evidence():
    truth = [{'kind': 'stump', 'x': 0, 'y': 0}, {'kind': 'stump', 'x': 10, 'y': 0},
             {'kind': 'log', 'x': 3, 'y': 0}, {'kind': 'bag', 'x': 0, 'y': 5}]
    items = [{'kind': 'stump', 'x': 0.5, 'y': 0}, {'kind': 'log', 'x': 4.5, 'y': 0},
             {'kind': 'bag', 'x': 0, 'y': 5.3}, {'kind': 'stump', 'x': 20, 'y': 20},
             {'kind': 'bag', 'x': 3, 'y': 0}]
    s = score_evidence(truth, items)
    assert s['stump'] == {'total': 2, 'found': 1, 'false': 1, 'mean_err_m': 0.5}
    assert s['log']['found'] == 1 and s['log']['false'] == 0
    assert s['bag']['found'] == 1 and s['bag']['false'] == 1


def test_box_distance_takes_object_not_background():
    depth = np.full((480, 720), 20.0, np.float32)
    depth[200:260, 300:360] = 4.0
    depth[0:10, 0:10] = np.inf
    assert abs(box_distance(depth, 295, 195, 365, 265, 0.5, 9.5) - 4.0) < 1e-6
    depth[200:260, 330:345] = 2.0
    assert abs(box_distance(depth, 295, 195, 365, 265, 0.5, 9.5,
                            exclude=[(328, 190, 347, 270)]) - 4.0) < 1e-6
    assert box_distance(depth, 295, 195, 365, 265, 0.5, 9.5) < 4.0
    far = np.full((480, 720), np.inf, np.float32)
    far[255:260, 300:360] = 7.0
    assert math.isnan(box_distance(far, 295, 245, 365, 262, 0.5, 9.5))
    assert math.isnan(box_distance(depth, 10, 300, 60, 350, 0.5, 9.5))


def test_pixel_ray_centre_and_left():
    x, y, z = pixel_ray(360, 240, 720, 480, 2.0944)
    assert (x, y, z) == (1.0, 0.0, 0.0)
    x, y, z = pixel_ray(0, 240, 720, 480, 2.0944)
    assert abs(math.degrees(math.atan2(y, x)) - 60.0) < 0.1


def test_tracker_merges_by_kind_and_distance():
    t = EvidenceTracker()
    a = t.add('stump', 0, 0, 0.9)
    assert t.add('stump', 0.5, 0, 0.6) is a and a.n == 2
    assert t.add('bag', 0.2, 0, 0.8) is not a
    assert t.add('stump', 3, 0, 0.8) is not a
    assert t.add('log', 0, 0, 0.7).id == 4
    assert 0 < a.x < 0.5 and a.conf == 0.9


def test_coordinator_verdict_uses_yolo_stumps_and_logs():
    from types import SimpleNamespace
    from deforestation_monitoring.mission_coordinator import MissionCoordinator
    node = SimpleNamespace(evidence=[('stump', 1.0, 1.0), ('log', 3.5, 0.0), ('bag', 1.0, 1.0),
                                     ('stump', 20.0, 20.0)])
    node._felled_evidence = lambda m: MissionCoordinator._felled_evidence(node, m)
    m = SimpleNamespace(box=(0.0, 0.0, 2.0, 2.0), x=1.0, y=1.0, type='CLEARING')
    assert sorted(MissionCoordinator._felled_evidence(node, m)) == ['log', 'stump']
    verdict, why = MissionCoordinator._classify_disturbance(node, m, '')
    assert verdict == 'CUT' and '1 stump, 1 felled log' in why
    node.evidence = []
    assert MissionCoordinator._classify_disturbance(node, m, '')[0] == 'UNCERTAIN'
