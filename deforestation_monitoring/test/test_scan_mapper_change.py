# test_scan_mapper_change.py: Scan_mapper change gating (GATE 1: a no-change run must emit nothing)
import numpy as np
import pytest
import rclpy

from deforestation_monitoring.scan_mapper import FILL_LOST_VALUE, ScanMapper


@pytest.fixture(scope="module")
def ros_ctx():
    rclpy.init()
    yield
    rclpy.shutdown()


def _mapper():
    return ScanMapper()


def _with_baseline(node, heights, hits, frac=None):
    node.height_grid[:] = heights
    node.hits_grid[:] = hits
    node.baseline_heights = heights.copy()
    node.baseline_hits = hits > 0
    node.baseline_hit_counts = hits.copy()
    node.baseline_canopy_frac = (
        frac if frac is not None
        else (heights > node.canopy_threshold).astype(np.float32))
    node.baseline_established = True


def test_edge_cell_seen_once_at_baseline_is_not_new_canopy(ros_ctx):
    node = _mapper()
    shape = (node.dim_x, node.dim_y)
    heights = np.zeros(shape, dtype=np.float32)
    hits = np.full(shape, 10.0, dtype=np.float32)
    edge = (slice(100, 104), slice(100, 104))
    hits[edge] = 1.0
    _with_baseline(node, heights, hits)
    node.height_recent[:] = 0.0
    node.height_recent[edge] = 3.9
    node.high_streak[edge] = 5
    _, _, gained = node._compute_change()
    assert not gained.any(), "thinly observed baseline cell must not gain"
    node.destroy_node()


def test_well_observed_clearing_is_still_lost(ros_ctx):
    node = _mapper()
    shape = (node.dim_x, node.dim_y)
    heights = np.zeros(shape, dtype=np.float32)
    crown = (slice(150, 160), slice(150, 160))
    heights[crown] = 5.5
    hits = np.full(shape, 10.0, dtype=np.float32)
    _with_baseline(node, heights, hits)
    node.height_recent[:] = heights
    node.height_recent[crown] = 0.2
    node.low_streak[crown] = 5
    _, lost, gained = node._compute_change()
    assert lost[crown].all()
    assert not gained.any()
    node.destroy_node()


def test_flickering_crown_edge_is_not_lost(ros_ctx):
    node = _mapper()
    shape = (node.dim_x, node.dim_y)
    heights = np.zeros(shape, dtype=np.float32)
    edge = (slice(40, 44), slice(40, 44))
    heights[edge] = 5.0
    hits = np.full(shape, 10.0, dtype=np.float32)
    frac = np.zeros(shape, dtype=np.float32)
    frac[edge] = 0.5
    _with_baseline(node, heights, hits, frac)
    node.height_recent[:] = 0.0
    node.low_streak[edge] = 5
    _, lost, _ = node._compute_change()
    assert not lost.any()
    node.destroy_node()


def _scan_msg(range_m):
    from sensor_msgs.msg import LaserScan
    scan = LaserScan()
    scan.angle_min = -0.05
    scan.angle_increment = 0.01
    scan.range_max = 40.0
    scan.ranges = [float(range_m)] * 11
    return scan


def test_ground_below_zero_overwrites_a_removed_tree(ros_ctx):
    from geometry_msgs.msg import Quaternion
    node = ScanMapper()
    node._odom_received = True
    node._odom_q = Quaternion(w=1.0)
    node.scan_callback(_scan_msg(4.0))
    ix = iy = int((0.0 - node.origin_x) / node.res)
    assert node.height_recent[ix, iy] == pytest.approx(6.0, abs=0.05)
    node.scan_callback(_scan_msg(10.2))
    assert node.height_recent[ix, iy] == 0.0
    assert node.low_streak[ix, iy] == 1
    node.destroy_node()


def test_height_drop_onto_a_neighbour_crown_is_marked_not_lost(ros_ctx):
    from deforestation_monitoring.scan_mapper import DROP_VALUE
    node = _mapper()
    shape = (node.dim_x, node.dim_y)
    heights = np.zeros(shape, dtype=np.float32)
    crown = (slice(60, 66), slice(60, 66))
    heights[crown] = 6.2
    hits = np.full(shape, 10.0, dtype=np.float32)
    _with_baseline(node, heights, hits)
    node.baseline_canopy_mean = heights.copy()
    node.height_recent[:] = heights
    node.height_recent[crown] = 5.0
    node.drop_streak[crown] = node.low_streak_threshold
    node.drop_m = 0.0
    grid, lost, _ = node._compute_change()
    assert not (grid == DROP_VALUE).any() and not lost.any()
    node.drop_m = 1.0
    grid, lost, _ = node._compute_change()
    assert (grid[crown] == DROP_VALUE).all()
    assert not lost.any() and DROP_VALUE > -50
    node.drop_streak[crown] = 1
    grid, _, _ = node._compute_change()
    assert not (grid == DROP_VALUE).any()
    node.destroy_node()


def test_cell_first_seen_after_the_snapshot_gets_a_baseline(ros_ctx):
    from geometry_msgs.msg import Quaternion
    ix = None
    for fill in (False, True):
        node = ScanMapper()
        node.baseline_fill = fill
        node._odom_received = True
        node._odom_q = Quaternion(w=1.0)
        shape = (node.dim_x, node.dim_y)
        _with_baseline(node, np.zeros(shape, np.float32),
                       np.zeros(shape, np.float32))
        node.baseline_canopy_mean = np.zeros(shape, np.float32)
        for _ in range(node.change_min_baseline_hits):
            node.scan_callback(_scan_msg(4.0))
        ix = iy = int((0.0 - node.origin_x) / node.res)
        hits_after = node.baseline_hit_counts[ix, iy]
        for _ in range(node.low_streak_threshold):
            node.scan_callback(_scan_msg(10.2))
        node.scan_callback(_scan_msg(10.2))
        grid, lost, _ = node._compute_change()
        if fill:
            assert hits_after == node.change_min_baseline_hits
            assert node.baseline_hit_counts[ix, iy] == node.change_min_baseline_hits
            assert node.baseline_canopy_frac[ix, iy] == pytest.approx(1.0)
            assert grid[ix, iy] == FILL_LOST_VALUE and -50 < FILL_LOST_VALUE < 0
            assert not lost.any()
        else:
            assert hits_after == 0 and not lost.any() and not grid.any()
        node.destroy_node()


def test_filled_cell_needs_canopy_on_every_scan(ros_ctx):
    from geometry_msgs.msg import Quaternion
    node = ScanMapper()
    node.baseline_fill = True
    node._odom_received = True
    node._odom_q = Quaternion(w=1.0)
    shape = (node.dim_x, node.dim_y)
    _with_baseline(node, np.zeros(shape, np.float32), np.zeros(shape, np.float32))
    node.baseline_canopy_mean = np.zeros(shape, np.float32)
    for r in (4.0, 10.2, 4.0):
        node.scan_callback(_scan_msg(r))
    ix = iy = int((0.0 - node.origin_x) / node.res)
    assert node.baseline_filled[ix, iy]
    assert node.baseline_canopy_frac[ix, iy] == pytest.approx(2 / 3)
    for _ in range(node.low_streak_threshold):
        node.scan_callback(_scan_msg(10.2))
    grid, lost, _ = node._compute_change()
    assert not lost.any() and not grid.any()
    node.fill_canopy_fraction = 0.6
    grid, lost, _ = node._compute_change()
    assert grid[ix, iy] == FILL_LOST_VALUE
    node.destroy_node()


def test_snapshot_waits_for_survey_loops(ros_ctx):
    from geometry_msgs.msg import Quaternion
    node = ScanMapper()
    node._odom_received = True
    node._odom_q = Quaternion(w=1.0)
    node.coverage_required = 0.0
    node.baseline_threshold = 1
    node.baseline_min_loops = 2
    node.loop_counter.loops = 1
    node.scan_callback(_scan_msg(4.0))
    assert not node.baseline_established
    node.loop_counter.loops = 2
    node.scan_callback(_scan_msg(4.0))
    assert node.baseline_established
    node.destroy_node()
