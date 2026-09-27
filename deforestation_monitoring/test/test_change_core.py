"""Unit tests for the ROS-free ChangeCore.

Run from the package root:  python3 -m pytest test/
"""
import numpy as np

try:
    from deforestation_monitoring.change_detector import ChangeCore, parse_survey_status
except ImportError:  # running next to the file
    from change_detector import ChangeCore, parse_survey_status


def forest(shape=(160, 160), trees=((60, 60), (60, 100), (100, 80)), r=8, h=6.0):
    """Ground at 0.1 m with round crowns of height h (grid rows/cols, 0.25 m cells)."""
    g = np.full(shape, 0.1, dtype=np.float32)
    rr, cc = np.mgrid[:shape[0], :shape[1]]
    for (r0, c0) in trees:
        g[(rr - r0) ** 2 + (cc - c0) ** 2 <= r * r] = h
    return g


def push_broom(grid, phases, spacing=6):
    """Keep only scan-line columns (col % spacing in phases); NaN elsewhere.

    Mimics the 2 Hz drone lidar at 3 m/s: one 0.25 m line every ~1.5 m,
    with each pass over an area landing at a different phase.
    """
    out = np.full_like(grid, np.nan)
    cols = [c for c in range(grid.shape[1]) if c % spacing in phases]
    out[:, cols] = grid[:, cols]
    return out


def lost_regions(core, grid, n):
    for _ in range(n):
        lost, _, pct, _ = core.update(grid)
    return core.regions(lost, pct)


def test_removed_tree_detected_after_persistence():
    core = ChangeCore(persistence_updates=3, min_cluster_cells=8)
    core.accumulate(forest())
    core.freeze()
    after = forest(trees=((60, 60), (60, 100)))          # tree at (100, 80) removed
    _, _, cl = lost_regions(core, after, 2)
    assert cl == [], 'must not fire before persistence is met'
    _, _, cl = lost_regions(core, after, 1)
    assert len(cl) == 1
    assert abs(cl[0]['row'] - 100) < 1 and abs(cl[0]['col'] - 80) < 1
    assert cl[0]['mean_pct'] > 95


def test_striped_push_broom_scans_give_one_region():
    """Removed crown seen only on scan lines is joined into ONE region."""
    core = ChangeCore(persistence_updates=1, gap_bridge_cells=8)
    core.accumulate(push_broom(forest(), phases={0, 2, 4}))       # baseline passes
    core.freeze()
    after = push_broom(forest(trees=((60, 60), (60, 100))), phases={2, 3})
    region, pct, cl = lost_regions(core, after, 1)
    assert len(cl) == 1
    assert abs(cl[0]['row'] - 100) < 2 and abs(cl[0]['col'] - 80) < 3
    assert cl[0]['region_cells'] > cl[0]['cells']   # gaps were filled
    assert region.sum() == cl[0]['region_cells'] and (pct[region] > 0).all()


def test_speckle_and_small_drops_ignored():
    core = ChangeCore(persistence_updates=1)
    base = forest()
    core.accumulate(base)
    core.freeze()
    cur = base.copy()
    cur[60, 60] = 0.1                    # isolated grazing-beam misreads,
    cur[62, 64] = 0.1                    # close enough to be bridged together
    cur[58:63, 98:103] = 4.5             # 1.5 m drop, still canopy
    region, _, cl = lost_regions(core, cur, 3)
    assert cl == [] and not region.any()


def test_unscanned_cells_never_count():
    core = ChangeCore(persistence_updates=1)
    base = forest()
    base[:, :40] = np.nan                # never scanned at baseline
    core.accumulate(base)
    core.freeze()
    cur = forest()
    cur[:, :40] = 0.1
    cur[100:120, 100:120] = np.nan       # not re-scanned yet
    lost, gained, *_ = core.update(cur)
    assert not lost.any() and not gained.any()


def test_baseline_is_running_max():
    core = ChangeCore()
    a = forest(trees=((60, 60),))
    b = np.full_like(a, np.nan)
    b[100:110, 100:110] = 7.0
    core.accumulate(a)
    core.accumulate(b)                   # later partial map must not erase a
    core.freeze()
    assert core.baseline[60, 60] == 6.0 and core.baseline[105, 105] == 7.0


def test_regrowth_reported_as_gained():
    core = ChangeCore(persistence_updates=1)
    core.accumulate(forest(trees=()))
    core.freeze()
    lost, gained, _, gain_pct = core.update(forest(trees=((80, 80),)))
    _, _, cl = core.regions(gained, gain_pct)
    assert len(cl) == 1 and not lost.any()


def test_survey_status_parser():
    assert parse_survey_status('STATE=PATROL wp=8/8 coverage=100% diverted=0') == (8, 8, 100.0)
    assert parse_survey_status('STATE=ORBITING target=(1.0,2.0)') is None
