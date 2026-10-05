"""Node-level gate tests for parrot_tree_tracker."""

import math

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
    """Regression: the freeze path runs the crown-coverage check, which
    uses scipy.ndimage. A missing import killed the node right after the
    coverage plateau (live sparse run stopped publishing at 99.8%)."""
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
    """Full coverage must score 1.0 (not ~0.6 from border erosion), a thin
    LiDAR stripe inside a flown-over crown is a scan gap, and a crown the
    drone only half flew over stays below the 0.8 admission gate."""
    node = ParrotTreeTracker()
    tree = DetectedTree(id=1, x=0.0, y=0.0, height=5.0, area_m2=3.0,
                        radius_m=1.0)
    ix = int((tree.x - node.origin_x) / node.res)
    iy = int((tree.y - node.origin_y) / node.res)

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
    """coverage_pct is a 0-1 fraction. Coverage still climbing ~1 pp per
    tick is NOT a plateau (the old 0.3 threshold = 30 pp said it was, and
    the baseline froze mid-survey at ~70%)."""
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
        node.coverage_pct = 0.60 + 0.01 * tick   # 60% -> 99%, still growing
        node.publish_status()
    assert node.baseline_frozen is False
    assert node._cov_stable_ticks < node.plateau_ticks
    # Coverage stops growing -> plateau -> freeze.
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
    """A 100%-scanned 0.25 m canopy map must fill the whole 0.2 m CHM.
    Forward mapping filled only 64% (an empty row/col every 5 cells), which
    showed up live as 'scan stripes' and was patched with offset terrain."""
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
    """Live RUN 3: oak_3's treetop jumped 1.84 m to another lobe of its own
    5 m crown and was reported GAINED. A detection inside a baseline crown
    is the same tree."""
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
    """The crown gate must not hide a genuinely new tree in open ground."""
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
    """Removal test 1 (dense): trees missing from the baseline flickered
    back into detection and were reported GAINED (#19–#25), although the
    canopy there never changed. A gain needs new-canopy evidence."""
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
    """cluster RUN 8: a 15 m crown collected change cells far away and
    fired LOST. Evidence must be searched near the tree only."""
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
    """Live RUN 5: candidates reset at the plateau, freeze fires on the 3rd
    tick; pine_2 missed one of those ticks, had only 2 hits, was frozen out
    and 6 s later reported GAINED. It must wait and enter the baseline."""
    node, frames, five = _five_tree_node(monkeypatch)
    without_pine = [d for d in five if (d.x, d.y) != (8, -8)]
    for dets in (five, five, without_pine, five):   # P, P+1, P+2, P+3
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


def test_drop_evidence_counts_only_when_enabled(ros_ctx, monkeypatch):
    """Height-drop cells (change map -30) are LOST evidence only with
    lost_drop_evidence; ground cells (-100) always are."""
    oak = {'id': 1, 'x': 0.0, 'y': 0.0, 'height': 5.6,
           'radius_m': 4.0, 'area_m2': 40.0}
    node = _frozen_node(monkeypatch, [oak, PINE], [
        DetectedTree(id=2, x=8.0, y=-8.0, height=3.1, area_m2=1.5,
                     radius_m=0.9)])
    _set_change(node, lost_at=(0.0, 0.0))
    node.change_dropped, node.change_lost = node.change_lost, \
        np.zeros_like(node.change_lost)
    assert node._lost_cells_near(oak, 2.0) == 0
    node.lost_drop_evidence = True
    assert node._lost_cells_near(oak, 2.0) >= node.lost_evidence_cells
    for _ in range(node.lost_streak_threshold + 2):
        node.publish_status()
    assert node.total_trees_lost == 1
    node.destroy_node()


def test_lost_needs_the_trees_own_top_to_drop(ros_ctx, monkeypatch):
    """Dense run 4: standing oak_64 (baseline position offset towards removed
    oak_78) collected 32 drop cells from oak_78's collapsing crown and was
    unmatched for 480 ticks. Its own top never dropped, so with
    lost_top_drop it must not be LOST; a tree whose top did drop is."""
    oak = {'id': 1, 'x': 0.0, 'y': 0.0, 'height': 6.2,
           'radius_m': 4.0, 'area_m2': 40.0}
    node = _frozen_node(monkeypatch, [oak, PINE], [
        DetectedTree(id=2, x=8.0, y=-8.0, height=3.1, area_m2=1.5,
                     radius_m=0.9)])
    node.lost_top_drop = 0.6
    shape = (node.dim_x, node.dim_y)
    ix = int((0.0 - node.origin_x) / node.res)
    top = (slice(ix - 2, ix + 3), slice(ix - 2, ix + 3))
    before = np.full(shape, 3.0, dtype=np.float32)
    before[top] = 6.2
    node.baseline_chm = before.copy()
    now = before.copy()                     # own top still there
    monkeypatch.setattr(node, "_fused_chm", lambda: (now, np.ones(shape, bool)))
    _set_change(node, lost_at=(0.0, 0.0))  # plenty of cell evidence nearby
    for _ in range(node.lost_streak_threshold + 2):
        node.publish_status()
    assert node.total_trees_lost == 0
    now[top] = 5.1                          # top gone (6.2 -> 5.1 m)
    for _ in range(node.lost_streak_threshold + 2):
        node.publish_status()
    assert node.total_trees_lost == 1
    node.destroy_node()


def test_lost_top_frac_needs_most_of_the_top_gone(ros_ctx, monkeypatch):
    """With lost_top_frac 0.9 a tree whose top cells mostly fell is LOST even
    if one neighbour cell keeps the disc max up (max condition relaxed to
    0.3 m); a tree with half its top standing is not."""
    oak = {'id': 1, 'x': 0.0, 'y': 0.0, 'height': 6.2,
           'radius_m': 4.0, 'area_m2': 40.0}
    node = _frozen_node(monkeypatch, [oak, PINE], [
        DetectedTree(id=2, x=8.0, y=-8.0, height=3.1, area_m2=1.5,
                     radius_m=0.9)])
    node.lost_top_drop, node.lost_top_frac = 0.3, 0.9
    shape = (node.dim_x, node.dim_y)
    ix = int((0.0 - node.origin_x) / node.res)
    before = np.full(shape, 3.0, dtype=np.float32)
    before[ix - 2:ix + 3, ix - 2:ix + 3] = 6.2          # 25 top cells
    node.baseline_chm = before.copy()
    now = before.copy()
    now[ix - 2:ix + 3, ix - 2:ix + 1] = 4.0             # 15/25 fell: 60 %
    monkeypatch.setattr(node, "_fused_chm", lambda: (now, np.ones(shape, bool)))
    _set_change(node, lost_at=(0.0, 0.0))
    for _ in range(node.lost_streak_threshold + 2):
        node.publish_status()
    assert node.total_trees_lost == 0
    now[ix - 2:ix + 3, ix - 2:ix + 3] = 4.0             # whole top gone ...
    now[ix + 2, ix + 2] = 5.9                           # ... bar one high cell
    for _ in range(node.lost_streak_threshold + 2):
        node.publish_status()
    assert node.total_trees_lost == 1
    node.destroy_node()


def _flicker_run(monkeypatch, **params):
    """pine_2 is seen for 5 ticks (jittering 0.3 m around x=8) and then
    drops out for the last 3 ticks before the freeze, like a thin pine top
    that a later pass reads lower (dense run 12, pine_75)."""
    node, frames, five = _five_tree_node(monkeypatch)
    for k, v in params.items():
        setattr(node, k, v)
    node._cov_stable_ticks = node.plateau_ticks - 7    # plateau on tick 7
    without_pine = [d for d in five if (d.x, d.y) != (8, -8)]
    for tick in range(12):
        if tick < 5:
            dx = 0.3 if tick % 2 == 0 else -0.3
            frames['dets'] = [DetectedTree(id=d.id, x=d.x + dx, y=d.y,
                                           height=d.height, area_m2=d.area_m2,
                                           radius_m=d.radius_m)
                              if (d.x, d.y) == (8, -8) else d for d in five]
        else:
            frames['dets'] = without_pine
        node.publish_status()
        if node.baseline_frozen:
            break
    return node


def test_majority_admission_keeps_a_flickering_pine(ros_ctx, monkeypatch):
    node = _flicker_run(monkeypatch, admit_window=8, baseline_stable_ticks=4,
                        candidate_max_misses=8,
                        reset_candidates_at_plateau=False,
                        baseline_mean_position=True)
    assert node.baseline_frozen and node.baseline_tree_count == 5
    pine = min(node.baseline_trees,
               key=lambda t: math.hypot(t['x'] - 8.0, t['y'] + 8.0))
    # averaged over the window: 8.06, not the last detection's 8.3
    assert abs(pine['x'] - 8.06) < 0.01 and abs(pine['y'] + 8.0) < 1e-6
    node.destroy_node()


def test_default_admission_drops_the_flickering_pine(ros_ctx, monkeypatch):
    node = _flicker_run(monkeypatch)
    assert node.baseline_frozen and node.baseline_tree_count == 4
    node.destroy_node()


def test_time_window_admission_keeps_a_flickering_pine(ros_ctx, monkeypatch):
    """Same flicker with a window in seconds: 5 hits in 8 ticks >= 50%."""
    node = _flicker_run(monkeypatch, admit_window_s=60.0, admit_fraction=0.5,
                        reset_candidates_at_plateau=False,
                        baseline_mean_position=True)
    assert node.baseline_frozen and node.baseline_tree_count == 5
    pine = min(node.baseline_trees,
               key=lambda t: math.hypot(t['x'] - 8.0, t['y'] + 8.0))
    assert abs(pine['x'] - 8.06) < 0.01
    node.destroy_node()


class _Clock:
    """Stand-in node clock so tests can step time in seconds."""

    def __init__(self, t=1000.0):
        self.t = t

    def now(self):
        return rclpy.time.Time(nanoseconds=int(self.t * 1e9))


def _colour_msg(spots, half=0.5):
    """A camera_species_mapper grid: {(x, y): value} squares, -1 elsewhere."""
    res, origin, dim = 0.25, -40.0, 320
    data = np.full((dim, dim), -1, dtype=np.int16)   # [row=y, col=x]
    for (x, y), value in spots.items():
        c0, c1 = int((x - half - origin) / res), int((x + half - origin) / res)
        r0, r1 = int((y - half - origin) / res), int((y + half - origin) / res)
        data[r0:r1, c0:c1] = int(round(value * 100))
    return _grid_msg(data, res, origin)


def _seen_node(monkeypatch, canopy=True, enabled=True):
    """PINE baselined as pine-coloured canopy, now undetected, no height evidence."""
    node = _frozen_node(monkeypatch, [PINE], [])
    node.baseline_trees[0]['camera_pine'] = canopy
    node.seen_lost = enabled
    clock = _Clock()
    monkeypatch.setattr(node, 'get_clock', lambda: clock)
    _set_change(node)                       # no loss cells anywhere
    return node, clock


def _ticks(node, clock, t, fraction=None, n=None):
    """Ticks at time t; the camera sees PINE's spot at `fraction` (None = not seen)."""
    clock.t = t
    spots = {} if fraction is None else {(PINE['x'], PINE['y']): fraction}
    node.seen_recent_cb(_colour_msg(spots))
    for _ in range(n or node.lost_streak_threshold + 1):
        node.publish_status()


def _visits(node, clock, fractions, t=1000.0):
    """One camera visit per fraction, 60 s apart; then a tick that ends the last."""
    for f in fractions:
        _ticks(node, clock, t, f)
        t += 60.0
    _ticks(node, clock, t)
    return t


def test_disc_mean_needs_enough_observed_cells(ros_ctx):
    node = ParrotTreeTracker()
    node.species_map_cb(_colour_msg({(8.0, -8.0): 0.70}, half=0.3))
    assert abs(node._disc_mean(node.species_mean, PINE) - 0.70) < 0.011
    node.seen_min_cells = 50
    assert node._disc_mean(node.species_mean, PINE) is None
    assert node._disc_mean(node.species_mean, {'x': 20.0, 'y': 20.0}) is None
    node.destroy_node()


def test_camera_lost_when_the_camera_sees_through_the_spot(ros_ctx, monkeypatch):
    """Dense pine_111: cut from under three oak crowns, its LiDAR top stays
    at 4.2 m (branch tips) and no ground shows, but the camera sees through
    the spot (canopy fraction 0.89 -> 0.00)."""
    node, clock = _seen_node(monkeypatch)
    _visits(node, clock, [0.0])
    assert node.total_trees_lost == 0       # one see-through visit is not enough
    _visits(node, clock, [0.0], t=1200.0)
    assert node.total_trees_lost == 1
    node.destroy_node()


def test_camera_judges_whole_visits_not_single_readings(ros_ctx, monkeypatch):
    """Standing pine_182 (baseline 1 m off its trunk): oblique readings at
    the crown edge dip to ~0.3 inside passes that average ~0.65."""
    node, clock = _seen_node(monkeypatch)
    t = 1000.0
    for _ in range(3):                      # three passes: low, high, low readings
        for f in (0.1, 0.9, 0.9, 0.1):
            _ticks(node, clock, t, f, n=2)
            t += 5.0
        t += 120.0
    _ticks(node, clock, t)
    assert node.total_trees_lost == 0
    node.destroy_node()


def test_canopy_visit_resets_the_camera_evidence(ros_ctx, monkeypatch):
    node, clock = _seen_node(monkeypatch)
    _visits(node, clock, [0.0, 0.9, 0.0])
    assert node.total_trees_lost == 0       # last two visits: 0.9 and 0.0
    _visits(node, clock, [0.0], t=1400.0)
    assert node.total_trees_lost == 1
    node.destroy_node()


def test_camera_lost_only_for_camera_pines_and_when_enabled(ros_ctx, monkeypatch):
    for canopy, enabled in ((False, True), (True, False)):
        node, clock = _seen_node(monkeypatch, canopy, enabled)
        _visits(node, clock, [0.0, 0.0, 0.0])
        assert node.total_trees_lost == 0
        node.destroy_node()


def test_camera_lost_also_needs_the_top_to_drop(ros_ctx, monkeypatch):
    """Replays of run 17 / gate3_dense: at phantom crowns (oak branch clumps)
    the camera sees through the sparse crown edge now and then although
    nothing was cut. The tree's own top must also have dropped."""
    node, clock = _seen_node(monkeypatch)
    node.lost_top_drop = 0.3
    shape = (node.dim_x, node.dim_y)
    ix = int((PINE['x'] - node.origin_x) / node.res)
    iy = int((PINE['y'] - node.origin_y) / node.res)
    top = (slice(ix - 2, ix + 3), slice(iy - 2, iy + 3))
    before = np.full(shape, 3.0, dtype=np.float32)
    before[top] = 4.7
    node.baseline_chm = before.copy()
    now = before.copy()                     # nothing dropped
    monkeypatch.setattr(node, "_fused_chm", lambda: (now, np.ones(shape, bool)))
    t = _visits(node, clock, [0.0, 0.0])
    assert node.total_trees_lost == 0
    now[top] = 4.2                          # pine_111: 4.7 -> 4.2 m
    _ticks(node, clock, t + 10.0)
    assert node.total_trees_lost == 1
    node.destroy_node()


def test_freeze_marks_pine_coloured_canopy_trees_for_the_camera(ros_ctx, monkeypatch):
    """Camera LOST applies only to trees the camera saw as canopy and as
    pine-coloured: every false see-through spot so far was an oak branch
    clump (blue/green 0.33-0.45), the cut pines read 0.71-0.74."""
    node = ParrotTreeTracker()
    chm = np.zeros((node.dim_x, node.dim_y), dtype=np.float32)
    scanned = np.ones((node.dim_x, node.dim_y), dtype=bool)
    monkeypatch.setattr(node, "_fused_chm", lambda: (chm, scanned))
    trees = [DetectedTree(id=1, x=0.0, y=0.0, height=6.0, area_m2=40.0, radius_m=3.5),
             DetectedTree(id=2, x=8.0, y=-8.0, height=4.5, area_m2=2.0, radius_m=0.9),
             DetectedTree(id=3, x=-8.0, y=8.0, height=4.5, area_m2=2.0, radius_m=0.9)]
    monkeypatch.setattr(
        "deforestation_monitoring.parrot_tree_tracker.detect_trees",
        lambda *a, **k: (list(trees), np.zeros(chm.shape, dtype=np.int32),
                         np.ones(chm.shape, dtype=bool)))
    # oak canopy / pine canopy / pine colour but hardly any canopy seen
    node.seen_map_cb(_colour_msg({(0.0, 0.0): 0.95, (8.0, -8.0): 0.95, (-8.0, 8.0): 0.2}))
    node.species_map_cb(_colour_msg({(0.0, 0.0): 0.39, (8.0, -8.0): 0.72, (-8.0, 8.0): 0.72}))
    node.coverage_pct = 0.998
    node._last_cov = 0.998
    node._cov_stable_ticks = 30
    for _ in range(node.baseline_stable_ticks + 1):
        node.publish_status()
    assert node.baseline_frozen
    labels = {(round(t['x']), round(t['y'])): t['camera_pine'] for t in node.baseline_trees}
    assert labels == {(0, 0): False, (8, -8): True, (-8, 8): False}
    node.destroy_node()


def test_colour_spires_only_until_the_baseline_freezes(ros_ctx, monkeypatch):
    """The colour map is an all-time mean: after pine_117 was cut it still
    read pine there, and a re-added tip kept the tree matched forever."""
    node = _frozen_node(monkeypatch, [PINE], [])
    node.colour_pines = True
    calls = []
    monkeypatch.setattr(node, '_colour_spires', lambda *a: calls.append(1) or [])
    node.publish_status()
    assert calls == []
    node.destroy_node()


def _spire_scene(node, pine_colour):
    """A 6 m oak crown at (0, 0) and a 4.8 m pine tip at (4, 0) beside it."""
    chm = np.zeros((node.dim_x, node.dim_y), dtype=np.float32)
    gx, gy = np.meshgrid(node.origin_x + (np.arange(node.dim_x) + 0.5) * node.res,
                         node.origin_y + (np.arange(node.dim_y) + 0.5) * node.res,
                         indexing='ij')
    r_oak = np.hypot(gx, gy)
    chm[r_oak <= 3.5] = 6.0 - 0.8 * r_oak[r_oak <= 3.5]
    r_pine = np.hypot(gx - 4.0, gy)
    chm = np.maximum(chm, np.where(r_pine <= 1.2, 4.8 - 1.5 * r_pine, 0.0)).astype(np.float32)
    node.species_map_cb(_colour_msg({(0.0, 0.0): 0.39, (4.0, 0.0): pine_colour}, half=0.6))
    return chm, np.ones(chm.shape, dtype=bool)


def test_colour_spire_adds_a_pine_the_height_detector_missed(ros_ctx):
    node = ParrotTreeTracker()
    chm, scanned = _spire_scene(node, 0.70)
    oak = DetectedTree(id=1, x=0.0, y=0.0, height=6.0, area_m2=60.0, radius_m=4.5)
    added = node._colour_spires(chm, scanned, [oak])
    assert len(added) == 1
    assert math.hypot(added[0].x - 4.0, added[0].y) < 0.3
    assert added[0].height > 4.5
    node.destroy_node()


def test_colour_spire_needs_pine_colour_and_no_detection_nearby(ros_ctx):
    node = ParrotTreeTracker()
    chm, scanned = _spire_scene(node, 0.40)           # oak-coloured tip: a clump
    oak = DetectedTree(id=1, x=0.0, y=0.0, height=6.0, area_m2=60.0, radius_m=4.5)
    assert node._colour_spires(chm, scanned, [oak]) == []
    chm, scanned = _spire_scene(node, 0.70)           # already detected nearby
    near = DetectedTree(id=2, x=3.2, y=0.4, height=4.6, area_m2=3.0, radius_m=1.0)
    assert node._colour_spires(chm, scanned, [oak, near]) == []
    node.destroy_node()


def test_baseline_detects_on_the_time_averaged_chm_until_the_freeze(ros_ctx, monkeypatch):
    """Dense runs 17-22: the latest-reading map reshapes crown edges on every
    pass, so pines dropped out of the baseline and oak_153 froze 2 m off.
    Before the freeze the detector sees the average; after it, the latest."""
    node = ParrotTreeTracker()
    node.baseline_mean_chm = True
    shape = (node.dim_x, node.dim_y)
    maps = [np.full(shape, 4.0, np.float32), np.full(shape, 6.0, np.float32)]
    tick = {'i': 0}
    monkeypatch.setattr(node, "_fused_chm",
                        lambda: (maps[tick['i'] % 2], np.ones(shape, bool)))
    seen = []
    monkeypatch.setattr(
        "deforestation_monitoring.parrot_tree_tracker.detect_trees",
        lambda chm, *a, **k: (seen.append(float(chm[0, 0])) or [],
                              np.zeros(shape, dtype=np.int32), np.ones(shape, dtype=bool)))
    for _ in range(3):
        node.publish_status()
        tick['i'] += 1
    assert seen == [4.0, 5.0, pytest.approx(14.0 / 3)]
    node.baseline_frozen = True
    node.publish_status()                   # tick 3: latest map is 6 m
    assert seen[-1] == 6.0
    node.destroy_node()


def _merge_node(monkeypatch):
    node = ParrotTreeTracker()
    node.merge_lost_crowns = True
    events = []
    monkeypatch.setattr(node, '_publish_event', lambda kind, tid, *a: events.append(tid))
    monkeypatch.setattr(node, '_publish_change_marker', lambda *a, **k: None)
    return node, events


OAK_153 = {'id': 27, 'x': 16.9, 'y': 20.7, 'height': 5.2, 'radius_m': 4.6, 'area_m2': 45.0, 'colour': 0.28}
CLUMP = {'id': 57, 'x': 13.3, 'y': 19.6, 'height': 4.7, 'radius_m': 1.5, 'area_m2': 6.0, 'colour': 0.34}


def test_clump_lost_with_its_oak_is_merged_either_order(ros_ctx, monkeypatch):
    """Dense replays: a clump of oak_153's crown went LOST 1-17 s before or
    after oak_153 and was scored as a double count."""
    for first, second in ((CLUMP, OAK_153), (OAK_153, CLUMP)):
        node, events = _merge_node(monkeypatch)
        node._pending_lost = [(100.0, first), (112.0, second)]
        node._flush_pending_lost(125.0)          # only the first is due
        node._flush_pending_lost(150.0)
        assert events == [27]
        assert node.total_trees_lost == 1
        node.destroy_node()


def test_clump_lost_well_after_its_oak_is_still_merged(ros_ctx, monkeypatch):
    """Live run 24: oak_153's clump went LOST 38 s after oak_153 itself,
    past the 30 s hold; replay of run 23: oak_60's clump 373 s after (next
    survey loop). A clump after its oak is merged up to merge_after_s."""
    for gap, merged in ((38.0, True), (373.0, True), (700.0, False)):
        node, events = _merge_node(monkeypatch)
        node._pending_lost = [(100.0, OAK_153)]
        node._flush_pending_lost(130.0)          # oak_153 reported
        node._pending_lost.append((100.0 + gap, CLUMP))
        node._flush_pending_lost(100.0 + gap + 30.0)
        assert events == ([27] if merged else [27, 57])
        node.destroy_node()


def test_later_half_of_a_split_crown_merges_into_a_smaller_host(ros_ctx, monkeypatch):
    """Replays 25 and 29 (lost_mean_drop): the baseline split oak_60's crown
    into #28 (16.5 m2, at the trunk) and #75 (17.8 m2, 3.2 m away); #75 went
    LOST 560 s after #28 and was not merged because it is the larger one.
    With merge_after_ratio a later crown merges into a reported host of at
    least that fraction of its area; held crowns still need a larger host."""
    host = {'id': 28, 'x': -16.5, 'y': -22.6, 'height': 6.0, 'area_m2': 16.5, 'colour': 0.30}
    half = {'id': 75, 'x': -13.8, 'y': -24.3, 'height': 5.9, 'area_m2': 17.8, 'colour': 0.31}
    for ratio, host_area, merged in ((0.0, 16.5, False), (0.8, 16.5, True), (0.8, 12.0, False)):
        node, events = _merge_node(monkeypatch)
        node.merge_after_ratio = ratio
        node._pending_lost = [(100.0, dict(host, area_m2=host_area))]
        node._flush_pending_lost(130.0)          # #28 reported
        node._pending_lost.append((660.0, half))
        node._flush_pending_lost(690.0)
        assert events == ([28] if merged else [28, 75])
        node.destroy_node()
    # both held at once: the smaller one is not a host for the larger one
    node, events = _merge_node(monkeypatch)
    node.merge_after_ratio = 0.8
    node._pending_lost = [(100.0, host), (110.0, half)]
    node._flush_pending_lost(150.0)
    assert sorted(events) == [75]                # #28 merged into the larger #75
    node.destroy_node()


def test_pines_and_lone_crowns_are_never_merged(ros_ctx, monkeypatch):
    pine = {'id': 60, 'x': 14.0, 'y': 19.0, 'height': 4.6, 'radius_m': 1.2, 'area_m2': 3.0,
            'colour': 0.73}
    node, events = _merge_node(monkeypatch)
    node._pending_lost = [(100.0, OAK_153), (105.0, pine)]
    node._flush_pending_lost(140.0)
    assert sorted(events) == [27, 60]          # a pine cut under the oak still counts
    node.destroy_node()
    node, events = _merge_node(monkeypatch)
    node._pending_lost = [(100.0, CLUMP)]      # its oak is still standing
    node._flush_pending_lost(140.0)
    assert events == [57]
    node.destroy_node()


def test_without_the_camera_merging_uses_the_top_height(ros_ctx, monkeypatch):
    oak = {k: v for k, v in OAK_153.items() if k != 'colour'}
    clump = {k: v for k, v in CLUMP.items() if k != 'colour'}
    node, events = _merge_node(monkeypatch)
    node._pending_lost = [(100.0, dict(oak, top=6.1)), (105.0, dict(clump, top=5.8))]
    node._flush_pending_lost(140.0)
    assert events == [27]                      # both oak-tall on the latest map
    node.destroy_node()
    node, events = _merge_node(monkeypatch)
    node._pending_lost = [(100.0, dict(oak, top=6.1)), (105.0, dict(clump, top=4.6))]
    node._flush_pending_lost(140.0)
    assert sorted(events) == [27, 57]          # pine-tall: kept
    node.destroy_node()


def test_pine_patches_only_where_the_baseline_has_no_tree(ros_ctx):
    node = ParrotTreeTracker()
    node.camera_area_alerts = True
    node.baseline_trees = [dict(PINE, x=10.0, y=10.0)]
    spots = {(5.0, 5.0): 0.72, (-5.0, -5.0): 0.39, (10.0, 10.0): 0.72}
    node.species_map_cb(_colour_msg(spots))
    node.seen_map_cb(_colour_msg({p: 0.9 for p in spots}))
    patches = node._find_pine_patches()
    assert [(round(p['x']), round(p['y'])) for p in patches] == [(5, 5)]
    node.destroy_node()


def test_pine_patch_alert_is_a_canopy_lost_event(ros_ctx, monkeypatch):
    """Crown-shared pine_117 is never in the baseline; the camera saw through
    its patch 0.91 -> 0.00 after the cut (runs 17, 19, 20)."""
    from deforestation_monitoring.removal_test import parse_canopy_events
    node, clock = _seen_node(monkeypatch, canopy=False)
    node.camera_area_alerts = True
    node._pine_patches = [{'id': 'P1', 'x': 4.3, 'y': -20.0, 'cells': 23}]
    sent = []
    monkeypatch.setattr(node.canopy_alert_pub, 'publish', lambda m: sent.append(m.data))
    t = 1000.0
    for fraction in (0.0, 0.0, None):
        clock.t = t
        spots = {} if fraction is None else {(4.3, -20.0): fraction}
        node.seen_recent_cb(_colour_msg(spots))
        for _ in range(3):
            node.publish_status()
        t += 60.0
    assert len(sent) == 1
    assert parse_canopy_events(sent[0]) == [('LOST', 4.3, -20.0)]
    assert node.total_trees_lost == 0        # an area alert, not a tree
    node.destroy_node()


def test_pine_patch_next_to_an_off_crown_tree_becomes_its_camera_site(ros_ctx):
    """Dense pine_163: baselined as #68 1.0 m off its trunk, where the disc
    reads a neighbour's colour (0.46); its own pine patch sat 0.9 m away and
    was skipped, so neither camera rule watched it (run 23)."""
    for attach, pine_near in ((True, False), (False, False), (True, True)):
        node = ParrotTreeTracker()
        node.camera_area_alerts = True
        node.area_attach = attach
        off = dict(PINE, id=68, x=20.8, y=-2.7, camera_pine=False, colour=0.46)
        trees = [off] + ([dict(PINE, id=69, x=21.8, y=-3.7, camera_pine=True)] if pine_near else [])
        node.baseline_trees = trees
        node.species_map_cb(_colour_msg({(20.8, -3.7): 0.72}))
        node.seen_map_cb(_colour_msg({(20.8, -3.7): 0.9}))
        assert node._find_pine_patches() == []
        site = off.get('site')
        if attach and not pine_near:
            assert math.hypot(site['x'] - 20.8, site['y'] + 3.7) < 0.2
            assert site['r'] == node.area_radius
            assert off['colour'] is None       # merging falls back to its top height
        else:
            assert site is None
        node.destroy_node()


def test_tree_with_a_camera_site_goes_lost_when_the_camera_sees_through_it(ros_ctx, monkeypatch):
    node, clock = _seen_node(monkeypatch, canopy=False)
    node.area_mean_drop = 0.0                # the height guard has its own test
    tree = node.baseline_trees[0]
    patch = {'x': PINE['x'], 'y': PINE['y'] - 1.0, 'r': 0.5}
    tree['site'] = dict(patch, id='P7', cells=25)
    t = 1000.0
    for fraction in (0.0, 0.0, None):
        clock.t = t
        spots = {} if fraction is None else {(patch['x'], patch['y']): fraction}
        node.seen_recent_cb(_colour_msg(spots, half=0.6))
        for _ in range(node.lost_streak_threshold + 1):
            node.publish_status()
        t += 60.0
    assert node.total_trees_lost == 1
    node.destroy_node()


def test_patch_loss_is_judged_on_the_disc_mean(ros_ctx, monkeypatch):
    """Crown-shared pine_117 (run 18): the camera saw through its patch
    twice, but an oak branch held the disc max (drop 0.0 m); the disc mean
    fell 2.6 m."""
    node, clock = _seen_node(monkeypatch, canopy=False)
    node.camera_area_alerts = True
    node.lost_top_drop = 0.3
    patch = {'id': 'P1', 'x': 4.3, 'y': -20.0, 'cells': 23, 'r': node.area_radius}
    node._pine_patches = [patch]
    shape = (node.dim_x, node.dim_y)
    before = np.full(shape, 4.4, dtype=np.float32)
    node.baseline_chm = before.copy()
    now = before.copy()
    monkeypatch.setattr(node, "_fused_chm", lambda: (now, np.ones(shape, bool)))
    sent = []
    monkeypatch.setattr(node.canopy_alert_pub, 'publish', lambda m: sent.append(m.data))

    def visit(t, fraction):
        clock.t = t
        spots = {} if fraction is None else {(4.3, -20.0): fraction}
        node.seen_recent_cb(_colour_msg(spots))
        for _ in range(3):
            node.publish_status()

    for i, fraction in enumerate((0.0, 0.0, None)):
        visit(1000.0 + 60.0 * i, fraction)
    assert sent == []                         # seen through, but no height loss
    ix = int((4.3 - node.origin_x) / node.res)
    iy = int((-20.0 - node.origin_y) / node.res)
    now[ix - 3:ix + 4, iy - 3:iy + 4] = 1.5   # the crown is gone ...
    now[ix + 1, iy] = 4.4                     # ... but for one branch tip
    visit(1200.0, None)
    assert len(sent) == 1
    node.destroy_node()


def test_pine_patch_the_lidar_also_saw_joins_the_baseline(ros_ctx):
    """pine_111 flickers on the pre-freeze map (live run 24: detected in 2
    of the last 30 ticks, not in the baseline); its 26-cell pine patch is
    admitted because the LiDAR saw a tree there. Sparse oak-edge patches
    are 6-8 cells and stay patches."""
    node = ParrotTreeTracker()
    node.camera_area_alerts = True
    node.baseline_trees = [dict(PINE, id=1)]
    pine = {'id': 'P1', 'x': -0.4, 'y': 19.5, 'cells': 26, 'r': 0.5}
    edge = {'id': 'P2', 'x': -2.9, 'y': 10.5, 'cells': 7, 'r': 0.5}
    node._pine_patches = [pine, edge]
    det = lambda x, y: DetectedTree(id=0, x=x, y=y, height=4.4, area_m2=3.0, radius_m=1.0)
    node._recent_dets.extend([[det(-0.3, 19.4)], [], [det(-2.8, 10.4)], []])
    admitted = node._admit_pine_patches()
    assert [(t['id'], round(t['x'], 1), t['hits']) for t in admitted] == [(2, -0.3, 1)]
    assert admitted[0]['camera_pine'] and admitted[0]['site'] is pine
    assert node._pine_patches == [edge]
    assert len(node.baseline_trees) == 2
    node.destroy_node()


def test_large_pine_patch_clear_of_every_tree_joins_as_camera_only(ros_ctx):
    """pine_111 missed the baseline in replays of runs 22 and 26 (no LiDAR
    hit near its 26-28 cell patch). Large patches clear of every baseline
    tree (2 m = the removal test's identity radius) join as camera-only."""
    node = ParrotTreeTracker()
    node.camera_area_alerts = True
    node.area_admit_cells = 0                 # the LiDAR-hit path is tested elsewhere
    node.camera_grid = (0.25, -40.0, -40.0)
    node.baseline_trees = [dict(PINE, id=1, x=10.0, y=10.0)]
    clear = {'id': 'P1', 'x': -0.4, 'y': 19.5, 'cells': 26, 'r': 0.5}
    near = {'id': 'P2', 'x': 11.8, 'y': 10.0, 'cells': 26, 'r': 0.5}     # 1.8 m from #1
    small = {'id': 'P3', 'x': -2.9, 'y': 10.5, 'cells': 8, 'r': 0.5}      # sparse oak edge
    node._pine_patches = [clear, near, small]
    admitted = node._admit_pine_patches()
    assert [(t['id'], t['camera_only'], t['site'] is clear) for t in admitted] == [(2, True, True)]
    assert node._pine_patches == [near, small]
    node.destroy_node()


def test_camera_only_trees_are_left_out_of_matching(ros_ctx):
    """A camera-only tree must not take a LiDAR detection or shrink its
    neighbours' match radius (0.4 x nearest-neighbour distance)."""
    node = ParrotTreeTracker()
    lidar = dict(PINE, id=1, x=0.0, y=0.0)
    cam = dict(PINE, id=2, x=2.0, y=0.0, camera_only=True)
    node.baseline_trees = [lidar, cam]
    det = DetectedTree(id=0, x=1.2, y=0.0, height=4.5, area_m2=3.0, radius_m=1.0)
    ids, unmatched = node._match_baseline([det])
    assert ids == [None] and unmatched == [0, 1]      # 1.2 m > the lone tree's track radius
    det = DetectedTree(id=0, x=0.3, y=0.0, height=4.5, area_m2=3.0, radius_m=1.0)
    ids, unmatched = node._match_baseline([det])
    assert ids == [1] and unmatched == [1]
    node.destroy_node()


def test_camera_only_tree_goes_lost_only_on_the_camera(ros_ctx, monkeypatch):
    node, clock = _seen_node(monkeypatch, canopy=True)
    node.baseline_trees[0]['camera_only'] = True
    _set_change(node, lost_at=(PINE['x'], PINE['y']))   # loss cells at it: no height LOST
    for _ in range(node.lost_streak_threshold + 2):
        node.publish_status()
    assert node.total_trees_lost == 0
    _visits(node, clock, [0.0, 0.0])
    assert node.total_trees_lost == 1
    node.destroy_node()



def test_baseline_pine_loss_counts_on_the_disc_mean_too(ros_ctx, monkeypatch):
    """Live run 29: pine_111 seen through from 252 s and its disc mean down
    3.0 m from 282 s, but a neighbour's branch held the disc max until
    667 s. A baseline pine's height check also passes on the mean."""
    node, clock = _seen_node(monkeypatch)
    node.lost_top_drop = 0.3
    shape = (node.dim_x, node.dim_y)
    ix = int((PINE['x'] - node.origin_x) / node.res)
    iy = int((PINE['y'] - node.origin_y) / node.res)
    before = np.full(shape, 1.0, dtype=np.float32)
    before[ix - 3:ix + 4, iy - 3:iy + 4] = 4.4
    node.baseline_chm = before.copy()
    now = before.copy()
    now[ix - 3:ix + 4, iy - 3:iy + 4] = 1.0           # the pine is gone ...
    now[ix + 2, iy] = 4.5                             # ... a branch tip holds the max
    monkeypatch.setattr(node, "_fused_chm", lambda: (now, np.ones(shape, bool)))
    _visits(node, clock, [0.0, 0.0])
    assert node.total_trees_lost == 1
    node.destroy_node()


def test_baseline_tree_enters_at_its_median_recent_position(ros_ctx):
    """Live run 28: oak_153 0.7-0.8 m from its trunk for 27 ticks, then
    1.8, 2.1, 2.1 m in the last three before the freeze."""
    node = ParrotTreeTracker()
    node.freeze_median_ticks = 10
    det = lambda x: DetectedTree(id=0, x=x, y=0.0, height=6.0, area_m2=40.0, radius_m=4.0)
    far = DetectedTree(id=0, x=9.0, y=0.0, height=6.0, area_m2=40.0, radius_m=4.0)
    for x in [0.8] * 7 + [1.8, 2.1, 2.1]:
        node._recent_dets.append([det(x), far])
    assert node._median_position({'x': 2.1, 'y': 0.0}) == (0.8, 0.0)
    assert node._median_position({'x': 9.0, 'y': 0.0}) == (9.0, 0.0)
    assert node._median_position({'x': 20.0, 'y': 0.0}) == (20.0, 0.0)   # never seen: unchanged
    node.destroy_node()


def test_median_skips_a_past_crown_shared_by_two_freeze_trees(ros_ctx):
    """Live run 40: oak_153 and oak_168 were detected as one fused crown at
    (17.8, 22.4) for most of the survey and as two trees at the freeze,
    (17.0, 21.0) and (19.4, 22.8). The 30-tick / 2 m median pulled both onto
    the fused spot. With freeze_median_exclusive a past detection within the
    radius of two freeze trees is skipped; a one-tree jump (run 28) still
    gets the median."""
    node = ParrotTreeTracker()
    node.freeze_median_ticks, node.freeze_median_radius = 30, 2.0
    det = lambda x, y: DetectedTree(id=0, x=x, y=y, height=6.0, area_m2=40.0, radius_m=3.5)
    for _ in range(25):
        node._recent_dets.append([det(17.8, 22.4)])
    for _ in range(3):
        node._recent_dets.append([det(17.0, 21.0), det(19.4, 22.8)])
    final = [(17.0, 21.0), (19.4, 22.8)]
    node.freeze_median_exclusive = False
    assert node._median_position({'x': 17.0, 'y': 21.0}, final) == (17.8, 22.4)
    node.freeze_median_exclusive = True
    assert node._median_position({'x': 17.0, 'y': 21.0}, final) == (17.0, 21.0)
    assert node._median_position({'x': 19.4, 'y': 22.8}, final) == (19.4, 22.8)
    # a single tree whose last ticks jumped is still pulled back
    node._recent_dets.clear()
    for x in [0.8] * 27 + [1.8, 2.1, 2.1]:
        node._recent_dets.append([det(x, 0.0)])
    assert node._median_position({'x': 2.1, 'y': 0.0}, [(2.1, 0.0)]) == (0.8, 0.0)
    node.destroy_node()


def test_clump_with_its_oak_still_standing_is_held_for_it(ros_ctx, monkeypatch):
    """Replays 25 / 28: oak_153's clump went LOST 172-189 s before oak_153.
    A small non-pine crown next to a larger standing one is held up to
    merge_hold_s; if the larger one goes LOST meanwhile, they merge."""
    node, _ = _merge_node(monkeypatch)
    node.merge_hold_s = 240.0
    target = dict(CLUMP, area_m2=18.0)        # oak_137, run 28: 18 m2 next to 45 m2
    node.baseline_trees = [dict(OAK_153), target]
    assert not node._has_standing_host(target)   # a real oak is not held
    node.destroy_node()
    for oak_goes, expected in ((True, [27]), (False, [57])):
        node, events = _merge_node(monkeypatch)
        node.merge_hold_s = 240.0
        oak, clump = dict(OAK_153), dict(CLUMP)
        node.baseline_trees = [oak, clump]
        node._alerted_lost_ids.add(57)
        clump['hold'] = 240.0 if node._has_standing_host(clump) else node.merge_window_s
        assert clump['hold'] == 240.0
        node._pending_lost = [(100.0, clump)]
        node._flush_pending_lost(130.0)
        assert events == []                       # still held
        if oak_goes:
            node._alerted_lost_ids.add(27)
            oak['hold'] = 240.0 if node._has_standing_host(oak) else node.merge_window_s
            node._pending_lost.append((270.0, oak))
            node._flush_pending_lost(300.0)
        node._flush_pending_lost(340.0)
        assert events == expected
        node.destroy_node()


def test_dashboard_notes_say_what_caught_a_lost_and_what_was_merged(ros_ctx, monkeypatch):
    import json
    node, events = _merge_node(monkeypatch)
    notes, infos = [], []
    monkeypatch.setattr(node.change_note_pub, 'publish', lambda m: notes.append(json.loads(m.data)))
    monkeypatch.setattr(node.baseline_info_pub, 'publish', lambda m: infos.append(json.loads(m.data)))
    oak, clump = dict(OAK_153), dict(CLUMP)
    cam = dict(PINE, id=90, camera_only=True, camera_pine=True, colour=0.72)
    node.baseline_trees = [oak, clump, cam]
    node._publish_baseline()
    kinds = {t['id']: (t['source'], t['species']) for t in infos[0]['trees']}
    assert kinds == {27: ('lidar', 'oak'), 57: ('lidar', 'oak'), 90: ('camera', 'pine')}
    oak['lost_by'] = 'height'
    node._pending_lost = [(100.0, oak), (105.0, clump)]
    node._flush_pending_lost(140.0)
    assert [(n['type'], n['id']) for n in notes] == [('LOST', 27), ('MERGED', 57)]
    assert notes[0]['by'] == 'height' and notes[1]['into'] == 27
    node.destroy_node()


def test_pine_patches_keep_off_the_survey_edge(ros_ctx):
    """Edge trees are tracked out to ±37 m, but a pine patch 1.7 m from the
    world edge raised a false camera alert: patches need area_edge_margin."""
    node = ParrotTreeTracker()
    node.camera_area_alerts = True
    node.survey_x_min = node.survey_y_min = -37.0
    node.survey_x_max = node.survey_y_max = 37.0
    node.baseline_trees = []
    spots = {(-35.3, -15.3): 0.72, (5.0, 5.0): 0.72}
    node.species_map_cb(_colour_msg(spots))
    node.seen_map_cb(_colour_msg({p: 0.9 for p in spots}))
    assert len(node._find_pine_patches()) == 2
    node.area_edge_margin = 7.0
    assert [(round(p['x']), round(p['y'])) for p in node._find_pine_patches()] == [(5, 5)]
    node.destroy_node()


def test_lost_height_rule_also_passes_on_the_disc_mean(ros_ctx, monkeypatch):
    """Run 34, oak_69: one branch tip held the disc max (6.2 -> 6.1 m) while
    the mean within 1 m fell 5.4 -> 2.5 m."""
    node = _frozen_node(monkeypatch, [OAK], [])
    node.lost_top_drop, node.lost_top_frac = 0.3, 0.9
    tree = node.baseline_trees[0]
    shape = (node.dim_x, node.dim_y)
    ix = int((tree['x'] - node.origin_x) / node.res)
    iy = int((tree['y'] - node.origin_y) / node.res)
    before = np.full(shape, 1.0, dtype=np.float32)
    before[ix - 5:ix + 6, iy - 5:iy + 6] = 6.0
    before[ix, iy] = 6.2
    node.baseline_chm = before
    now = before.copy()
    now[ix - 5:ix + 6, iy - 5:iy + 6] = 2.5          # the crown is gone ...
    now[ix + 1, iy] = 6.1                            # ... but for one branch tip
    node._current_chm = now
    assert not node._top_fell(tree)                  # max fell 0.1 m only
    node.lost_mean_drop = 2.0
    assert node._top_fell(tree)
    node._current_chm = before.copy()                # standing: nothing fell
    assert not node._top_fell(tree)
    node.destroy_node()
