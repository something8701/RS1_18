"""Node-level gate tests for parrot_tree_tracker."""

import numpy as np
import pytest
import rclpy

from deforestation_monitoring.parrot_tree_tracker import ParrotTreeTracker
from deforestation_monitoring.tree_detection import DetectedTree


@pytest.fixture(scope="module")
def ros_ctx():
    rclpy.init()
    yield
    rclpy.shutdown()


def test_no_change_events_before_baseline_freeze(ros_ctx, monkeypatch):
    """Before the baseline freezes, lost/gained must stay zero and no
    change event may be evaluated, even when detections already exist."""
    node = ParrotTreeTracker()
    node.baseline_frozen = False
    node.drone_baseline_ready = False

    fake_chm = np.zeros((4, 4), dtype=np.float32)
    fake_scanned = np.ones((4, 4), dtype=bool)
    monkeypatch.setattr(node, "_fused_chm",
                        lambda: (fake_chm, fake_scanned))
    monkeypatch.setattr(
        "deforestation_monitoring.parrot_tree_tracker.detect_trees",
        lambda *a, **k: ([DetectedTree(
            id=1, x=0.0, y=0.0, height=5.0, area_m2=2.0, radius_m=1.0)],
            np.zeros((4, 4), dtype=np.int32),
            np.ones((4, 4), dtype=bool)))

    node.publish_status()

    assert node.baseline_frozen is False
    assert node.total_trees_lost == 0
    assert node.total_trees_gained == 0
    assert node.baseline_tree_count == 0
    node.destroy_node()


def test_baseline_freeze_after_plateau_does_not_crash(ros_ctx, monkeypatch):
    """The freeze runs the crown coverage check, which uses scipy.ndimage.
    A missing import used to crash the node right after the coverage
    plateau."""
    node = ParrotTreeTracker()
    chm = np.zeros((node.dim_x, node.dim_y), dtype=np.float32)
    scanned = np.ones((node.dim_x, node.dim_y), dtype=bool)
    monkeypatch.setattr(node, "_fused_chm", lambda: (chm, scanned))
    trees = [
        DetectedTree(id=1, x=0.0, y=0.0, height=5.0, area_m2=2.0,
                     radius_m=1.0),
        DetectedTree(id=2, x=5.0, y=5.0, height=6.0, area_m2=3.0,
                     radius_m=1.2),
    ]
    monkeypatch.setattr(
        "deforestation_monitoring.parrot_tree_tracker.detect_trees",
        lambda *a, **k: (list(trees),
                         np.zeros(chm.shape, dtype=np.int32),
                         np.ones(chm.shape, dtype=bool)))

    # Coverage already on its plateau.
    node.coverage_pct = 0.998
    node._last_cov = 0.998
    node._cov_stable_ticks = 30

    for _ in range(node.baseline_stable_ticks + 1):
        node.publish_status()

    assert node.baseline_frozen is True
    assert node.baseline_tree_count == 2
    assert node.total_trees_lost == 0
    assert node.total_trees_gained == 0
    node.destroy_node()


def test_freeze_blocked_reason_is_complete(ros_ctx):
    """The blocked message must carry the counts, not a truncated prefix."""
    node = ParrotTreeTracker()
    tiny = DetectedTree(id=1, x=0.0, y=0.0, height=5.0, area_m2=0.1,
                        radius_m=0.3)
    ok, reason = node._freeze_baseline([tiny], None)
    assert ok is False
    assert "0/1 candidates" in reason
    node.destroy_node()


def test_crown_observed_fraction(ros_ctx):
    """Full coverage must score 1.0 (the border must not erode), a thin
    missing stripe inside a crown the drone flew over is a scan gap, and a
    crown the drone only half flew over stays below the 0.8 admission level."""
    node = ParrotTreeTracker()
    tree = DetectedTree(id=1, x=0.0, y=0.0, height=5.0, area_m2=3.0,
                        radius_m=1.0)
    ix = int((tree.x - node.origin_x) / node.res)

    full = np.ones((node.dim_x, node.dim_y), dtype=bool)
    assert node._neighborhood_observed_fraction(tree, full) == 1.0

    stripe = full.copy()
    stripe[ix, :] = False  # one missing scan line through the crown
    assert node._neighborhood_observed_fraction(tree, stripe) == 1.0

    half = full.copy()
    half[:ix, :] = False   # drone never flew over the lower half
    assert node._neighborhood_observed_fraction(tree, half) < 0.8
    node.destroy_node()


def test_plateau_needs_coverage_to_stop_growing(ros_ctx, monkeypatch):
    """coverage_pct is a 0-1 fraction. Coverage that still grows about one
    percentage point per tick is not a plateau."""
    node = ParrotTreeTracker()
    chm = np.zeros((node.dim_x, node.dim_y), dtype=np.float32)
    scanned = np.ones((node.dim_x, node.dim_y), dtype=bool)
    monkeypatch.setattr(node, "_fused_chm", lambda: (chm, scanned))
    monkeypatch.setattr(
        "deforestation_monitoring.parrot_tree_tracker.detect_trees",
        lambda *a, **k: ([DetectedTree(id=1, x=0.0, y=0.0, height=5.0,
                                       area_m2=2.0, radius_m=1.0),
                          DetectedTree(id=2, x=5.0, y=5.0, height=5.0,
                                       area_m2=2.0, radius_m=1.0)],
                         np.zeros(chm.shape, dtype=np.int32),
                         np.ones(chm.shape, dtype=bool)))
    for tick in range(40):
        node.coverage_pct = 0.60 + 0.01 * tick   # 60% to 99%, still growing
        node.publish_status()
    assert node.baseline_frozen is False
    assert node._cov_stable_ticks < node.plateau_ticks
    # Coverage stops growing, so it plateaus and the baseline freezes.
    for _ in range(node.plateau_ticks + node.baseline_stable_ticks + 1):
        node.publish_status()
    assert node.baseline_frozen is True
    node.destroy_node()


def test_freeze_requires_min_baseline_trees(ros_ctx):
    """One admissible tree must not freeze a baseline that needs two."""
    node = ParrotTreeTracker()
    scanned = np.ones((node.dim_x, node.dim_y), dtype=bool)
    good = DetectedTree(id=1, x=0.0, y=0.0, height=5.0, area_m2=2.0,
                        radius_m=1.0)
    tiny = DetectedTree(id=2, x=5.0, y=5.0, height=5.0, area_m2=0.1,
                        radius_m=0.3)
    ok, reason = node._freeze_baseline([good, tiny], scanned)
    assert ok is False
    assert node.baseline_frozen is False
    assert "1/2 candidates" in reason
    node.destroy_node()


def _grid_msg(data_yx, res, origin):
    from nav_msgs.msg import OccupancyGrid
    msg = OccupancyGrid()
    msg.info.resolution = res
    msg.info.height, msg.info.width = data_yx.shape
    msg.info.origin.position.x = origin
    msg.info.origin.position.y = origin
    msg.data = data_yx.astype(np.int8).flatten().tolist()
    return msg


def test_fused_chm_fills_grid_from_coarser_map(ros_ctx):
    """A fully scanned 0.25 m canopy map must fill the whole 0.2 m CHM.
    Mapping forward filled only 64% of it (an empty row or column every 5
    cells)."""
    node = ParrotTreeTracker()
    data = np.zeros((320, 320), dtype=np.int16)   # [y, x], 0 m everywhere
    # One tall cell (5.0 m) at world x=10.1, y=-5.1.
    col = int((10.1 + 40.0) / 0.25)
    row = int((-5.1 + 40.0) / 0.25)
    data[row, col] = 50
    node.canopy_cb(_grid_msg(data, 0.25, -40.0))
    node.terrain_hits[:] = 0
    fused, scanned = node._fused_chm()
    assert scanned.mean() == 1.0
    ix = int((10.1 - node.origin_x) / node.res)
    iy = int((-5.1 - node.origin_y) / node.res)
    assert fused[ix, iy] == 5.0, "tall cell must land at (x, y), not (y, x)"
    assert fused[iy, ix] == 0.0
    node.destroy_node()


def _frozen_node(monkeypatch, baseline, detections):
    node = ParrotTreeTracker()
    chm = np.zeros((node.dim_x, node.dim_y), dtype=np.float32)
    scanned = np.ones((node.dim_x, node.dim_y), dtype=bool)
    monkeypatch.setattr(node, "_fused_chm", lambda: (chm, scanned))
    monkeypatch.setattr(
        "deforestation_monitoring.parrot_tree_tracker.detect_trees",
        lambda *a, **k: (list(detections),
                         np.zeros(chm.shape, dtype=np.int32),
                         np.ones(chm.shape, dtype=bool)))
    node.baseline_trees = [dict(t) for t in baseline]
    node.baseline_tree_count = len(baseline)
    node.next_tree_id = len(baseline) + 1
    node.baseline_scanned = scanned.copy()
    node.baseline_frozen = True
    return node


OAK = {'id': 1, 'x': -0.7, 'y': 7.9, 'height': 5.6,
       'radius_m': 5.1, 'area_m2': 51.0}
PINE = {'id': 2, 'x': 8.0, 'y': -8.0, 'height': 3.1,
        'radius_m': 0.9, 'area_m2': 1.5}


def test_lobe_jump_inside_crown_is_not_a_gain(ros_ctx, monkeypatch):
    """An oak's treetop moved 1.84 m to another lobe of its own 5 m crown.
    A detection inside a baseline crown is the same tree, not a gain."""
    lobe = DetectedTree(id=1, x=-2.0, y=9.2, height=5.4, area_m2=20.0,
                        radius_m=3.0)
    pine = DetectedTree(id=2, x=8.0, y=-8.0, height=3.1, area_m2=1.5,
                        radius_m=0.9)
    node = _frozen_node(monkeypatch, [OAK, PINE], [lobe, pine])
    for _ in range(node.gain_stable_ticks + 2):
        node.publish_status()
    assert node.total_trees_gained == 0
    assert node._gains_rejected_in_crown >= 1
    node.destroy_node()


def test_new_tree_outside_crowns_is_still_gained(ros_ctx, monkeypatch):
    """The crown check must not hide a new tree in open ground."""
    oak = DetectedTree(id=1, x=-0.7, y=7.9, height=5.6, area_m2=51.0,
                       radius_m=5.1)
    pine = DetectedTree(id=2, x=8.0, y=-8.0, height=3.1, area_m2=1.5,
                        radius_m=0.9)
    new = DetectedTree(id=3, x=-10.0, y=-10.0, height=3.0, area_m2=1.5,
                       radius_m=0.8)
    node = _frozen_node(monkeypatch, [OAK, PINE], [oak, pine, new])
    _set_change(node, gained_at=(-10.0, -10.0))
    for _ in range(node.gain_stable_ticks + 2):
        node.publish_status()
    assert node.total_trees_gained == 1
    node.destroy_node()


def _set_change(node, gained_at=None, lost_at=None, half=0.5):
    """Fake /canopy_change_map cells (0.25 m grid, origin -40)."""
    res, origin, dim = 0.25, -40.0, 320
    gained = np.zeros((dim, dim), dtype=bool)   # [row=y, col=x]
    lost = np.zeros((dim, dim), dtype=bool)
    for grid, at in ((gained, gained_at), (lost, lost_at)):
        if at is None:
            continue
        c0 = int((at[0] - half - origin) / res)
        c1 = int((at[0] + half - origin) / res)
        r0 = int((at[1] - half - origin) / res)
        r1 = int((at[1] + half - origin) / res)
        grid[r0:r1, c0:c1] = True
    node.change_gained, node.change_lost = gained, lost
    node.change_res, node.change_ox, node.change_oy = res, origin, origin


def test_flickering_detection_without_new_canopy_is_not_a_gain(
        ros_ctx, monkeypatch):
    """Trees missing from the baseline that flicker back into detection are
    not gains while the canopy there has not changed. A gain needs
    new-canopy evidence."""
    oak = DetectedTree(id=1, x=-0.7, y=7.9, height=5.6, area_m2=51.0,
                       radius_m=5.1)
    pine = DetectedTree(id=2, x=8.0, y=-8.0, height=3.1, area_m2=1.5,
                        radius_m=0.9)
    flicker = DetectedTree(id=3, x=-10.0, y=-10.0, height=5.0, area_m2=20.0,
                           radius_m=4.0)
    node = _frozen_node(monkeypatch, [OAK, PINE], [oak, pine, flicker])
    _set_change(node)            # change map: no new canopy anywhere
    for _ in range(node.gain_stable_ticks + 2):
        node.publish_status()
    assert node.total_trees_gained == 0
    assert node._gains_rejected_no_evidence >= 1
    node.destroy_node()


def test_lost_evidence_radius_is_capped(ros_ctx, monkeypatch):
    """A 15 m crown must not collect change cells far away and go LOST.
    Evidence is only counted near the tree."""
    big = {'id': 1, 'x': 0.0, 'y': 0.0, 'height': 5.6,
           'radius_m': 15.0, 'area_m2': 159.0}
    node = _frozen_node(monkeypatch, [big, PINE], [
        DetectedTree(id=2, x=8.0, y=-8.0, height=3.1, area_m2=1.5,
                     radius_m=0.9)])
    _set_change(node, lost_at=(9.0, 9.0))     # 12.7 m away
    for _ in range(node.lost_streak_threshold + 2):
        node.publish_status()
    assert node.total_trees_lost == 0
    node.destroy_node()


def _five_tree_node(monkeypatch):
    node = ParrotTreeTracker()
    chm = np.zeros((node.dim_x, node.dim_y), dtype=np.float32)
    scanned = np.ones((node.dim_x, node.dim_y), dtype=bool)
    monkeypatch.setattr(node, "_fused_chm", lambda: (chm, scanned))
    five = [DetectedTree(id=i, x=float(x), y=float(y), height=5.0, area_m2=2.0,
                         radius_m=1.0)
            for i, (x, y) in enumerate(
                [(-8, -8), (8, -8), (0, 8), (-12, 0), (12, 0)], start=1)]
    frames = {'dets': five}
    monkeypatch.setattr(
        "deforestation_monitoring.parrot_tree_tracker.detect_trees",
        lambda *a, **k: (list(frames['dets']),
                         np.zeros(chm.shape, dtype=np.int32),
                         np.ones(chm.shape, dtype=bool)))
    node.coverage_pct = 0.99
    node._last_cov = 0.99
    node._cov_stable_ticks = node.plateau_ticks - 1   # plateau next tick
    return node, frames, five


def test_one_tick_dropout_at_freeze_is_not_left_out(ros_ctx, monkeypatch):
    """Candidates reset at the plateau and the freeze can happen on the 3rd
    tick. A tree that misses one of those ticks must be waited for and enter
    the baseline, not be reported GAINED later."""
    node, frames, five = _five_tree_node(monkeypatch)
    without_pine = [d for d in five if (d.x, d.y) != (8, -8)]
    for dets in (five, five, without_pine, five):   # plateau tick P to P+3
        frames['dets'] = dets
        node.publish_status()
    assert node.baseline_frozen is True
    assert node.baseline_tree_count == 5
    frames['dets'] = five
    for _ in range(node.gain_stable_ticks + 2):
        node.publish_status()
    assert node.total_trees_gained == 0
    node.destroy_node()


def test_freeze_does_not_wait_forever_for_a_flickering_candidate(
        ros_ctx, monkeypatch):
    """A candidate that never reaches 3-of-4 must not block the baseline
    beyond freeze_max_wait_ticks."""
    node, frames, five = _five_tree_node(monkeypatch)
    without_pine = [d for d in five if (d.x, d.y) != (8, -8)]
    for tick in range(node.freeze_max_wait_ticks + 2):
        # pine_2 seen every other tick: never 3 of the last 4
        frames['dets'] = five if tick % 2 == 0 else without_pine
        node.publish_status()
        if node.baseline_frozen:
            break
    assert node.baseline_frozen is True
    assert node.baseline_tree_count == 4
    node.destroy_node()
