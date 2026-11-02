# test_drone_prior.py: Drone_prior_map: the drone's trees as a cost mask for the Husky's planner
import math

import numpy as np
from nav_msgs.msg import OccupancyGrid

from deforestation_monitoring.area_search import free_spot
from deforestation_monitoring.drone_prior_map import canopy_top, classify, prior_mask

RES, O, N = 0.25, -10.0, 80
CORE = {'pine': 2.2, 'oak': 2.1}


def _at(mask, x, y):
    return int(mask[int((y - O) / RES), int((x - O) / RES)])


def test_disc_is_full_cost_to_the_core_then_fades_to_zero():
    m = prior_mask([(0.0, 0.0, 'pine')], RES, O, O, N, N, CORE, 0.8, 85.0)
    assert _at(m, 0.1, 0.1) == 85
    assert _at(m, 2.0, 0.1) == 85
    mid = _at(m, 2.6 + 0.01, 0.12)
    assert 30 <= mid <= 55
    assert _at(m, 3.2, 0.1) == 0
    assert m.min() == 0 and m.max() == 85


def test_species_set_the_core_and_overlaps_keep_the_higher_cost():
    pine = prior_mask([(0.0, 0.0, 'pine')], RES, O, O, N, N, CORE, 0.8, 85.0)
    oak = prior_mask([(0.0, 0.0, 'oak')], RES, O, O, N, N, CORE, 0.8, 85.0)
    assert (pine >= oak).all() and (pine > oak).any()
    both = prior_mask([(0.0, 0.0, 'pine'), (3.5, 0.0, 'pine')], RES, O, O, N, N, CORE, 0.8, 85.0)
    assert _at(both, 1.75, 0.1) == 85
    assert (both >= pine).all()


def test_trees_at_or_beyond_the_edge_do_not_break_the_grid():
    m = prior_mask([(9.9, 9.9, 'oak'), (50.0, 50.0, 'pine')], RES, O, O, N, N, CORE, 0.8, 85.0)
    assert m.shape == (N, N) and _at(m, 9.8, 9.8) == 85
    assert (m[:40, :40] == 0).all()


def test_coordinator_moves_a_waypoint_out_of_a_pine():
    m = prior_mask([(0.0, 0.0, 'pine')], RES, O, O, N, N, CORE, 0.8, 85.0)
    assert free_spot(m, RES, O, O, 0.3, 0.0, max_shift=1.5) is None
    spot = free_spot(m, RES, O, O, 1.5, 0.0, max_shift=1.5)
    assert spot is not None and math.hypot(*spot) >= CORE['pine']


def test_species_fall_back_on_the_canopy_height():
    assert classify('oak', 4.0, 5.4) == 'oak'
    assert classify(None, 6.1, 5.4) == 'oak'
    assert classify(None, 4.6, 5.4) == 'pine'
    assert classify(None, None, 5.4) == 'pine'


def test_canopy_top_reads_decimetres_around_the_point():
    g = OccupancyGrid()
    g.info.resolution, g.info.width, g.info.height = RES, N, N
    g.info.origin.position.x = g.info.origin.position.y = O
    d = np.full((N, N), -1, np.int8)
    d[40, 44] = 61
    g.data = d.ravel().tolist()
    assert canopy_top(g, 1.0, 0.0) == 6.1
    assert canopy_top(g, -5.0, -5.0) is None
    assert canopy_top(None, 0.0, 0.0) is None


import json

import pytest


@pytest.fixture(scope='module')
def ros_ctx():
    import rclpy
    rclpy.init()
    yield
    rclpy.shutdown()


def test_p12_a_new_baseline_clears_the_cut_trees(ros_ctx):
    from std_msgs.msg import String
    from deforestation_interfaces.msg import TreeChangeEvent
    from deforestation_monitoring.drone_prior_map import DronePriorMap
    n = DronePriorMap()
    try:
        base = {'trees': [{'id': 1, 'x': 0.0, 'y': 0.0, 'species': 'pine'},
                          {'id': 2, 'x': 5.0, 'y': 0.0, 'species': 'oak'}]}
        n.baseline_cb(String(data=json.dumps(base)))
        n.event_cb(TreeChangeEvent(event_type='LOST', tree_id=1))
        assert n.cut == {1}
        n.baseline_cb(String(data=json.dumps(base)))
        assert n.cut == {1}
        reset = {'trees': [{'id': 1, 'x': 9.0, 'y': 9.0, 'species': 'pine'}]}
        n.baseline_cb(String(data=json.dumps(reset)))
        assert n.cut == set() and set(n.trees) == {1}
    finally:
        n.destroy_node()
