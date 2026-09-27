"""scan_mapper change checks: a run with no changes must report nothing."""

import numpy as np
import pytest
import rclpy

from deforestation_monitoring.scan_mapper import ScanMapper


@pytest.fixture(scope="module")
def ros_ctx():
    rclpy.init()
    yield
    rclpy.shutdown()


def _mapper():
    return ScanMapper()   # default 0.5 m grid (160x160) is enough here


def _with_baseline(node, heights, hits, frac=None):
    """Fake a baseline. ``hits`` = scans that saw each cell, ``frac`` = the
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
    # A later lane sees the crown there again and again.
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
    """Crown-edge cells read canopy on some beams and ground on others. A
    cell that read canopy in only half its baseline scans is not 'lost'
    after a run of ground readings."""
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
    """Bare ground projects to about -0.2 m (fixed 10 m altitude vs the real
    sensor height). Ground must still overwrite the cell, otherwise a removed
    tree's height stays in /forest_canopy_map."""
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
