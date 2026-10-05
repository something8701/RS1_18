"""scan_mapper change gating (GATE 1: a no-change run must emit nothing)."""

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
    return ScanMapper()   # default 0.5 m grid (160x160) is enough here


def _with_baseline(node, heights, hits, frac=None):
    """Fake a baseline: ``hits`` = scans that touched each cell, ``frac`` =
    fraction of them that read canopy (default: from ``heights``)."""
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
    hits[edge] = 1.0            # grazing beam hit ground past the crown edge
    _with_baseline(node, heights, hits)
    # Later lane sees the crown there, repeatedly.
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
    node.height_recent[crown] = 0.2     # tree removed
    node.low_streak[crown] = 5
    _, lost, gained = node._compute_change()
    assert lost[crown].all()
    assert not gained.any()
    node.destroy_node()


def test_flickering_crown_edge_is_not_lost(ros_ctx):
    """Removal test 4: crown-edge cells read canopy on some beams and
    ground on others. Comparing the latest reading with the all-time max
    flagged 27/47 standing dense trees far from any removal. A cell that
    read canopy in only half its baseline scans is not 'lost' after a
    ground streak."""
    node = _mapper()
    shape = (node.dim_x, node.dim_y)
    heights = np.zeros(shape, dtype=np.float32)
    edge = (slice(40, 44), slice(40, 44))
    heights[edge] = 5.0                       # all-time max: canopy
    hits = np.full(shape, 10.0, dtype=np.float32)
    frac = np.zeros(shape, dtype=np.float32)
    frac[edge] = 0.5                          # canopy on half the scans
    _with_baseline(node, heights, hits, frac)
    node.height_recent[:] = 0.0
    node.low_streak[edge] = 5
    _, lost, _ = node._compute_change()
    assert not lost.any()
    node.destroy_node()


def _scan_msg(range_m):
    from sensor_msgs.msg import LaserScan
    scan = LaserScan()
    scan.angle_min = -0.05          # middle beam points straight down
    scan.angle_increment = 0.01
    scan.range_max = 40.0
    scan.ranges = [float(range_m)] * 11
    return scan


def test_ground_below_zero_overwrites_a_removed_tree(ros_ctx):
    """Removal tests 1-3: bare ground projects to z ~ -0.2 m (fixed 10 m
    altitude vs the real sensor height). With the old `max >= 0` touch
    rule, ground never overwrote a cell, so a removed tree's height stayed
    in /forest_canopy_map forever."""
    from geometry_msgs.msg import Quaternion
    node = ScanMapper()
    node._odom_received = True
    node._odom_q = Quaternion(w=1.0)
    node.scan_callback(_scan_msg(4.0))      # tree top at 6.0 m
    ix = iy = int((0.0 - node.origin_x) / node.res)
    assert node.height_recent[ix, iy] == pytest.approx(6.0, abs=0.05)
    node.scan_callback(_scan_msg(10.2))     # tree gone: ground at -0.2 m
    assert node.height_recent[ix, iy] == 0.0
    assert node.low_streak[ix, iy] == 1
    node.destroy_node()


def test_height_drop_onto_a_neighbour_crown_is_marked_not_lost(ros_ctx):
    """Dense removal 8: oak_153 fell onto its neighbours' crowns (6.2 ->
    5.1 m over the trunk), exposing almost no ground. With drop_evidence_m
    those cells are marked DROP_VALUE — distinct from -100 (lost to ground),
    and above pattern_scanner's -50 flag threshold."""
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
    node.height_recent[crown] = 5.0            # neighbour crown underneath
    node.drop_streak[crown] = node.low_streak_threshold
    node.drop_m = 0.0                          # off: nothing marked
    grid, lost, _ = node._compute_change()
    assert not (grid == DROP_VALUE).any() and not lost.any()
    node.drop_m = 1.0
    grid, lost, _ = node._compute_change()
    assert (grid[crown] == DROP_VALUE).all()
    assert not lost.any() and DROP_VALUE > -50
    node.drop_streak[crown] = 1                # not persistent yet
    grid, _, _ = node._compute_change()
    assert not (grid == DROP_VALUE).any()
    node.destroy_node()


def test_cell_first_seen_after_the_snapshot_gets_a_baseline(ros_ctx):
    """Edge trees, run 35: the snapshot comes at 90 % of the coverage box,
    before the drone flies its last lane, so a tree in that lane had no
    baseline scans and its removal left no lost cells. With baseline_fill a
    cell under change_min_baseline_hits keeps adding scans to its baseline
    until it has enough; then a later ground streak is a loss."""
    from geometry_msgs.msg import Quaternion
    ix = None
    for fill in (False, True):
        node = ScanMapper()
        node.baseline_fill = fill
        node._odom_received = True
        node._odom_q = Quaternion(w=1.0)
        shape = (node.dim_x, node.dim_y)
        _with_baseline(node, np.zeros(shape, np.float32),
                       np.zeros(shape, np.float32))      # nothing seen yet
        node.baseline_canopy_mean = np.zeros(shape, np.float32)
        for _ in range(node.change_min_baseline_hits):
            node.scan_callback(_scan_msg(4.0))           # tree top at 6.0 m
        ix = iy = int((0.0 - node.origin_x) / node.res)
        hits_after = node.baseline_hit_counts[ix, iy]
        for _ in range(node.low_streak_threshold):
            node.scan_callback(_scan_msg(10.2))          # tree cut: ground
        node.scan_callback(_scan_msg(10.2))              # one more: stays sealed
        grid, lost, _ = node._compute_change()
        if fill:
            assert hits_after == node.change_min_baseline_hits
            assert node.baseline_hit_counts[ix, iy] == node.change_min_baseline_hits
            assert node.baseline_canopy_frac[ix, iy] == pytest.approx(1.0)
            # tracker evidence, but no CANOPY LOST alert / pattern flag
            assert grid[ix, iy] == FILL_LOST_VALUE and -50 < FILL_LOST_VALUE < 0
            assert not lost.any()
        else:
            assert hits_after == 0 and not lost.any() and not grid.any()
        node.destroy_node()


def test_filled_cell_needs_canopy_on_every_scan(ros_ctx):
    """Fill replays (2026-10-05): a filled cell seals after 3 scans, so a
    flickering crown edge (canopy on 2 of 3) passed the 0.6 canopy fraction
    and gave false area alerts and a standing oak LOST. Filled cells need
    fill_canopy_fraction (1.0) of their scans to read canopy."""
    from geometry_msgs.msg import Quaternion
    node = ScanMapper()
    node.baseline_fill = True
    node._odom_received = True
    node._odom_q = Quaternion(w=1.0)
    shape = (node.dim_x, node.dim_y)
    _with_baseline(node, np.zeros(shape, np.float32), np.zeros(shape, np.float32))
    node.baseline_canopy_mean = np.zeros(shape, np.float32)
    for r in (4.0, 10.2, 4.0):                       # canopy, ground, canopy
        node.scan_callback(_scan_msg(r))
    ix = iy = int((0.0 - node.origin_x) / node.res)
    assert node.baseline_filled[ix, iy]
    assert node.baseline_canopy_frac[ix, iy] == pytest.approx(2 / 3)
    for _ in range(node.low_streak_threshold):
        node.scan_callback(_scan_msg(10.2))
    grid, lost, _ = node._compute_change()
    assert not lost.any() and not grid.any()
    node.fill_canopy_fraction = 0.6                  # the first version
    grid, lost, _ = node._compute_change()
    assert grid[ix, iy] == FILL_LOST_VALUE
    node.destroy_node()


def test_snapshot_waits_for_survey_loops(ros_ctx):
    """baseline_min_loops (dense world 2): coverage alone no longer
    freezes the change baseline; the patrol's loop count must also reach it."""
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
