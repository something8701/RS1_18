"""Detector-independent visibility classification."""

from pathlib import Path

import numpy as np

from deforestation_monitoring.removal_test import select_removal_targets
from deforestation_monitoring.tree_detection import parse_tree_truth
from deforestation_monitoring.visibility import (
    classify_tree, classify_trees, species_top_heights)

DATA = Path(__file__).parent / "data"
WORLDS = Path(__file__).resolve().parents[2] / "41068_ignition_bringup" / "worlds"
ORIGIN = (-40.0, -40.0)
RES = 0.25


def _map(name):
    g = np.load(DATA / name)["data"].astype(np.float32)
    return np.where(g >= 0, g / 10.0, -1.0)


def _grid(value=0.0):
    return np.full((320, 320), value, dtype=np.float32)


def _put(g, x, y, h, r=1.0):
    xs = ORIGIN[0] + (np.arange(320) + 0.5) * RES
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    g[(gx - x) ** 2 + (gy - y) ** 2 <= r * r] = h


def test_synthetic_classes():
    g = _grid()
    _put(g, 0, 0, 4.6)                 # pine top seen
    _put(g, 10, 0, 5.8, r=3.0)         # oak crown covering a pine
    g[:40, :40] = -1                   # unscanned corner
    assert classify_tree(g, RES, ORIGIN, ("pine_1", 0, 0))[0] == "visible"
    assert classify_tree(g, RES, ORIGIN, ("pine_2", 10, 0))[0] == "understory"
    assert classify_tree(g, RES, ORIGIN, ("pine_3", -37, -37))[0] == "unobserved"
    assert classify_tree(g, RES, ORIGIN, ("oak_4", 20, 20))[0] == "unobserved"


def test_species_heights_from_real_sparse_map():
    g = _map("sparse_live_canopy_map.npz")
    truth = parse_tree_truth(str(WORLDS / "sparse_trees.sdf"))
    h = species_top_heights(g, RES, ORIGIN, truth)
    assert abs(h["oak"] - 6.2) < 0.3 and abs(h["pine"] - 4.6) < 0.4


def test_dense_removal_targets_classification():
    """Validated on dense removal tests 4/6/8: pine_53 is understory (5.1-
    5.5 m over its trunk, unchanged after removal); the other 9 visible."""
    g = _map("dense_live_canopy_map.npz")
    truth = parse_tree_truth(str(WORLDS / "dense_forest.sdf"))
    targets = select_removal_targets(truth, 10, (-30.0, 30.0, -30.0, 30.0))
    cls = classify_trees(g, RES, ORIGIN, targets)
    assert cls["pine_53"][0] == "understory"
    assert all(c == "visible" for n, (c, _) in cls.items() if n != "pine_53")


def test_removal_tiers_for_dense_targets():
    from deforestation_monitoring.visibility import removal_tier
    g = _map("dense_live_canopy_map.npz")
    truth = parse_tree_truth(str(WORLDS / "dense_forest.sdf"))
    targets = select_removal_targets(truth, 10, (-30.0, 30.0, -30.0, 30.0))
    cls = classify_trees(g, RES, ORIGIN, targets)
    tiers = {t[0]: removal_tier(t, cls[t[0]][0], truth) for t in targets}
    assert tiers["pine_53"] == "understory"
    assert all(tiers[n] == "canopy" for n in tiers if n.startswith("oak"))
    assert tiers["pine_117"] == "crown-shared"      # oak 4.6 m away
    assert tiers["pine_163"] == "canopy"            # nearest oak 6.0 m


def test_showcase_trees_are_all_canopy_tier():
    """Option-B world: no pine within 7 m of an oak."""
    from deforestation_monitoring.visibility import removal_tier
    truth = parse_tree_truth(str(WORLDS / "showcase_forest.sdf"))
    assert all(removal_tier(t, "visible", truth) == "canopy" for t in truth)


def test_top_drop_cells_mode_ignores_a_single_high_cell():
    """Live dense run 9: pine_163's crown fell (74 cells > 0.8 m within
    2 m) but one 4.6 m cell kept the disc max unchanged. 'cells' re-reads
    only the cells that formed the top at freeze."""
    from deforestation_monitoring.visibility import top_drop
    res, org = 0.25, (-5.0, -5.0)
    before = np.full((40, 40), 3.6, dtype=np.float32)
    before[19:22, 19:22] = 4.6                       # the pine's top at (0, 0)
    now = np.full((40, 40), 3.6, dtype=np.float32)   # pine removed ...
    now[20, 17] = 4.6                                # ... one overhang cell
    assert top_drop(before, now, res, org, 0.0, 0.0, 1.0, 'max') < 0.1
    assert top_drop(before, now, res, org, 0.0, 0.0, 1.0, 'cells') >= 0.9
    # a standing tree: nothing changed
    assert top_drop(before, before.copy(), res, org, 0.0, 0.0, 1.0,
                    'cells') == 0.0
    # unscanned now -> unknown (0)
    assert top_drop(before, np.full_like(before, -1.0), res, org, 0.0, 0.0,
                    1.0, 'cells') == 0.0


def test_top_drop_mean_mode_sees_a_cut_crown_under_an_overhang():
    """Dense pine_163 (run 20): a neighbour's branch held the 0.5 m disc
    max at 4.3 m (4.4 before) while the disc mean fell 1.7 m."""
    from deforestation_monitoring.visibility import top_drop
    res, org = 0.25, (-5.0, -5.0)
    before = np.full((40, 40), 4.4, dtype=np.float32)
    now = np.full((40, 40), 1.0, dtype=np.float32)   # the crown is gone ...
    now[21, 20] = 4.3                                # ... but for one branch tip
    assert top_drop(before, now, res, org, 0.0, 0.0, 0.5, 'max') < 0.2
    assert top_drop(before, now, res, org, 0.0, 0.0, 0.5, 'mean') > 2.5
    assert top_drop(before, before.copy(), res, org, 0.0, 0.0, 0.5, 'mean') == 0.0
    assert top_drop(before, np.full_like(before, -1.0), res, org, 0.0, 0.0,
                    0.5, 'mean') == 0.0
