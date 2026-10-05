"""Offline tests for the CHM-based individual tree detector.

These run without ROS or Gazebo: the detector is exercised on synthetic
canopy surfaces built from known ground truth, which is exactly the
auto-calibration workflow the live node uses.
"""

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deforestation_monitoring.tree_detection import (  # noqa: E402
    DetectionParams,
    calibrate,
    detect_trees,
    otsu_threshold,
    parse_tree_truth,
    refine_canopy_mask,
    scenario_layouts,
    score_detections,
    synthetic_chm,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORLDS_DIR = PROJECT_ROOT / "41068_ignition_bringup" / "worlds"

ORIGIN = (-12.0, -12.0)
RES = 0.2
DIM = (120, 120)


def make_chm(truth, crown_factor=0.16, crown_offset=0.40, noise=0.04,
             ground_relief=0.4):
    chm = synthetic_chm(
        truth, RES, ORIGIN[0], ORIGIN[1], DIM[0], DIM[1],
        crown_factor=crown_factor, crown_offset=crown_offset,
        noise_sigma=noise, ground_relief=ground_relief)
    return chm, np.ones(chm.shape, dtype=bool)


def run(truth, params=None, **overrides):
    cfg = (params or DetectionParams(chm_resolution=RES))
    for key, value in overrides.items():
        setattr(cfg, key, value)
    chm, scanned = make_chm(truth)
    dets = detect_trees(chm, scanned, cfg, ORIGIN[0], ORIGIN[1])
    return dets, score_detections(dets, truth, match_radius=1.0)


def test_otsu_separates_ground_from_canopy():
    heights = np.concatenate([
        np.random.default_rng(0).uniform(0.0, 0.4, 4000),  # ground
        np.random.default_rng(1).uniform(4.0, 9.0, 6000),  # canopy
    ])
    threshold = otsu_threshold(heights)
    assert 0.4 < threshold < 4.0


def test_mask_refinement_removes_rays_keeps_small_crowns():
    """Shape filtering must drop 1-cell-wide edge rays without a hit-count
    threshold, while a compact crown seen only once survives."""
    mask = np.zeros((30, 30), dtype=bool)
    mask[14:17, 14:17] = True      # small compact crown
    mask[20, 2:10] = True          # 1-cell-wide ray streak
    params = DetectionParams(mask_open_iterations=1,
                             mask_close_iterations=2,
                             min_crown_cells=3)
    out = refine_canopy_mask(mask, params)
    assert out[15, 15], "small crown must survive filtering"
    assert not out[20, 2:10].any(), "1-cell-wide ray must be removed"


def test_isolated_trees_are_all_found():
    truth = scenario_layouts()["isolated_5"]
    _, metrics = run(truth)
    assert metrics["true_positives"] == 5
    assert metrics["false_positives"] == 0
    assert metrics["missed"] == 0
    assert metrics["f1"] == 1.0


def test_uneven_ground_is_never_labelled_as_tree():
    """Ground that reads 0.0-0.5 m must not produce phantom trees."""
    truth = scenario_layouts()["isolated_5"]
    chm, scanned = make_chm(truth, ground_relief=0.6)
    cfg = DetectionParams(chm_resolution=RES)
    dets = detect_trees(chm, scanned, cfg, ORIGIN[0], ORIGIN[1])
    metrics = score_detections(dets, truth, match_radius=1.0)
    assert metrics["true_positives"] == 5
    assert metrics["false_positives"] == 0
    assert metrics["f1"] == 1.0


def test_lop_sided_crown_is_one_tree_with_its_full_shape():
    """A crown is not a circle: an elongated, lop-sided crown must stay one
    tree and keep its full extent."""
    chm = np.zeros(DIM, dtype=np.float32)
    gx, gy = np.meshgrid(
        np.arange(DIM[0]) * RES,
        np.arange(DIM[1]) * RES,
        indexing="ij")
    # Elliptical crown centred at (0, 0), stretched along x.
    crown = 7.0 * np.exp(
        -(((gx - 6.0) / 4.0) ** 2 + ((gy - 6.0) / 1.5) ** 2) / 2.0)
    chm = crown
    scanned = np.ones(DIM, dtype=bool)
    cfg = DetectionParams(chm_resolution=RES)
    dets = detect_trees(chm, scanned, cfg, ORIGIN[0], ORIGIN[1])
    assert len(dets) == 1
    assert dets[0].area_m2 > 15.0  # full ellipse, not a small circle


def test_pit_inside_crown_does_not_split_tree():
    """A dropout pit inside a crown must be filled, not split the tree."""
    gx, gy = np.meshgrid(
        np.arange(DIM[0]) * RES,
        np.arange(DIM[1]) * RES,
        indexing="ij")
    chm = 7.0 * np.exp(
        -(((gx - 12.0) / 3.0) ** 2 + ((gy - 12.0) / 3.0) ** 2) / 2.0)
    chm[60, 60] = 1.0  # pit at the crown top
    scanned = np.ones(DIM, dtype=bool)
    cfg = DetectionParams(chm_resolution=RES, pit_fill_size=3)
    dets = detect_trees(chm, scanned, cfg, ORIGIN[0], ORIGIN[1])
    assert len(dets) == 1, "a crown with an internal pit is one tree"


@pytest.mark.parametrize("scenario", ["pair_2.5m"])
def test_resolvable_clusters_are_split(scenario):
    """Trees separated by >= ~1.5 m must stay individuals, not merge."""
    truth = scenario_layouts()[scenario]
    dets, metrics = run(truth)
    assert metrics["detected"] == len(truth), (
        f"{scenario}: expected {len(truth)} detections, got "
        f"{metrics['detected']} (TP={metrics['true_positives']}, "
        f"FP={metrics['false_positives']}, missed={metrics['missed']})")
    assert metrics["f1"] == 1.0


@pytest.mark.parametrize("scenario", [
    "pair_1.2m", "pair_1.8m", "triplet_1.5m",
])
def test_overlapping_crowns_stay_one_object(scenario):
    """When crowns physically overlap, Yang et al.'s greedy overlap selection
    keeps one object instead of inventing extra treetops."""
    truth = scenario_layouts()[scenario]
    dets, metrics = run(truth)
    assert metrics["detected"] == 1
    assert metrics["false_positives"] == 0


def test_tight_quad_partially_splits_without_false_positives():
    """A 1.3 m quad has fully overlapping crowns; the shape-aware watershed
    treats it as one canopy object and never invents extra treetops."""
    truth = scenario_layouts()["quad_1.3m"]
    dets, metrics = run(truth)
    assert 1 <= metrics["detected"] <= 2
    assert metrics["false_positives"] == 0


def test_mixed_scenario_high_recall():
    truth = scenario_layouts()["mixed"]
    _, metrics = run(truth)
    assert metrics["precision"] >= 0.8
    assert metrics["recall"] >= 0.4
    assert metrics["f1"] >= 0.6


def test_sdf_ground_truth_parsing():
    assert len(parse_tree_truth(str(WORLDS_DIR / "sparse_trees.sdf"))) == 5
    assert len(parse_tree_truth(str(WORLDS_DIR / "simple_trees.sdf"))) == 13
    assert len(parse_tree_truth(str(WORLDS_DIR / "dense_forest.sdf"))) == 214
    assert len(parse_tree_truth(str(WORLDS_DIR / "cluster_test.sdf"))) == 10
    assert len(parse_tree_truth(str(WORLDS_DIR / "showcase_forest.sdf"))) == 101


def test_calibration_recovers_high_f1_on_dense_clusters():
    truth = parse_tree_truth(str(WORLDS_DIR / "dense_forest.sdf"))
    # dense_forest spans roughly -37..37 m, so use the full 80 m grid.
    chm = synthetic_chm(truth, RES, -40.0, -40.0, 400, 400,
                        noise_sigma=0.04)
    scanned = np.ones(chm.shape, dtype=bool)
    result = calibrate(
        chm, scanned, truth,
        origin_x=-40.0, origin_y=-40.0,
        match_radius=1.0, n_trials=24, seed=3)
    assert result["best"]["metrics"] is not None
    assert result["best"]["metrics"]["f1"] >= 0.8


def test_detection_params_round_trip():
    original = DetectionParams(window_scale=0.17, min_crown_cells=4)
    restored = DetectionParams.from_dict(original.to_dict())
    assert restored == original


def test_crown_lobe_cannot_absorb_neighbouring_pine():
    """Live sparse bag, map 110: oak_5 split into its crown plus a lobe
    6.97 m from pine_2. With merge before overlap suppression the lobe
    absorbed the pine (merge_radius 7.0). The lobe must be suppressed
    first, and the pine kept."""
    from deforestation_monitoring.tree_detection import (
        DetectedTree, select_crowns)
    params = DetectionParams(overlap_threshold=0.25, merge_small_area=2.5,
                             merge_radius=7.0)
    oak = DetectedTree(id=1, x=12.39, y=-0.02, height=5.20, area_m2=30.38,
                       radius_m=4.59)
    lobe = DetectedTree(id=2, x=9.25, y=-1.14, height=5.15, area_m2=23.56,
                        radius_m=6.95)
    pine = DetectedTree(id=3, x=7.99, y=-8.00, height=3.13, area_m2=2.00,
                        radius_m=0.90)
    kept = select_crowns([oak, lobe, pine], params)
    positions = {(round(d.x, 2), round(d.y, 2)) for d in kept}
    assert (7.99, -8.0) in positions, "real pine must survive"
    assert (9.25, -1.14) not in positions, "lobe must be suppressed"
    assert len(kept) == 2


def _lobed_oak_chm():
    """Broad 5 m-radius crown over the trunk at (0, 0) plus a narrower,
    slightly taller lobe 2.5 m off-centre — the Fuel oak shape that put
    oak_3's treetop 2.55 m from its trunk in live sparse RUN 6."""
    gx, gy = np.meshgrid(
        ORIGIN[0] + (np.arange(DIM[0]) + 0.5) * RES,
        ORIGIN[1] + (np.arange(DIM[1]) + 0.5) * RES, indexing="ij")
    crown = 5.3 * np.exp(-(gx ** 2 + gy ** 2) / (2 * 2.2 ** 2))
    lobe = 5.6 * np.exp(-((gx + 2.0) ** 2 + (gy - 1.5) ** 2) / (2 * 0.9 ** 2))
    chm = np.maximum(crown, lobe).astype(np.float32)
    chm[chm < 1.0] = 0.0
    return chm, np.ones(DIM, dtype=bool)


def test_lobed_oak_is_one_tree_over_its_trunk():
    chm, scanned = _lobed_oak_chm()
    cfg = DetectionParams(chm_resolution=RES, overlap_threshold=0.25,
                          position_mode="crown")
    dets = detect_trees(chm, scanned, cfg, ORIGIN[0], ORIGIN[1])
    assert len(dets) == 1
    assert math.hypot(dets[0].x, dets[0].y) < 1.0, (
        f"tree at ({dets[0].x:.2f}, {dets[0].y:.2f}), trunk at (0, 0)")


def _live_sparse_map():
    """A real /forest_canopy_map from live sparse RUN 4 (map 18 of the
    recorded bag): oak_3's crown has an off-centre lobe."""
    f = np.load(Path(__file__).parent / "data" / "sparse_live_canopy_map.npz")
    g = f["data"].astype(np.float32)          # [ix, iy], -1 = unscanned
    scanned = g >= 0
    return np.where(scanned, g / 10.0, 0.0).astype(np.float32), scanned


def _shipped_params(**overrides):
    import yaml
    cfg = Path(__file__).resolve().parents[1] / "config" / \
        "tree_detection_params.yaml"
    base = yaml.safe_load(cfg.read_text())["/**"]["ros__parameters"]
    params = DetectionParams.from_dict(base)
    params.chm_resolution = 0.25
    for key, value in overrides.items():
        setattr(params, key, value)
    return params


def test_live_map_all_five_trees_within_half_a_metre():
    """Regression on real data: with lobe merging + crown position every
    sparse tree is found and within 0.5 m of its SDF trunk."""
    chm, scanned = _live_sparse_map()
    truth = parse_tree_truth(str(WORLDS_DIR / "sparse_trees.sdf"))
    dets = detect_trees(chm, scanned, _shipped_params(), -40.0, -40.0)
    metrics = score_detections(dets, truth, match_radius=0.5)
    assert (metrics["true_positives"], metrics["false_positives"],
            metrics["missed"]) == (5, 0, 0)


PRE_TUNING = dict(smooth_sigma=0.5, window_scale=0.35, window_offset=0.3,
                  max_window=3.0, min_sep=1.0, saddle_ratio=0.6,
                  overlap_threshold=0.25, merge_small_area=2.5,
                  min_crown_cells=10, ground_margin=0.8,
                  auto_threshold_cap=3.0)


def test_live_map_band_position_puts_oak_on_its_lobe():
    """Documents the failure "crown" fixed for the pre-tuning parameters:
    on the same map the top-band position put oak_3 ~2.5 m from its trunk
    (a miss at 1 m). The tuned parameters (smoothing 0.9) no longer split
    the lobe, see test_live_map_all_five_trees_within_half_a_metre."""
    chm, scanned = _live_sparse_map()
    dets = detect_trees(
        chm, scanned, _shipped_params(position_mode="band", **PRE_TUNING),
        -40.0, -40.0)
    oak_3 = min(dets, key=lambda d: math.hypot(d.x - 0.0, d.y - 8.0))
    assert math.hypot(oak_3.x, oak_3.y - 8.0) > 1.0


def test_params_yaml_round_trip_with_strings():
    import yaml
    from deforestation_monitoring.tree_detection import params_to_ros_yaml
    p = DetectionParams(position_mode="band", min_crown_cells=7,
                        use_auto_threshold=False)
    loaded = yaml.safe_load(params_to_ros_yaml(p))["/**"]["ros__parameters"]
    assert DetectionParams.from_dict(loaded) == p


def test_touching_labels():
    from deforestation_monitoring.tree_detection import _touching_labels
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[1:4, 1:4] = 1          # isolated
    labels[6:9, 1:4] = 2          # touches 3
    labels[6:9, 4:7] = 3
    assert _touching_labels(labels) == {2, 3}


def test_auto_position_keeps_isolated_sparse_oaks_on_the_trunk():
    """Fixed-mapper sparse maps: top-band position put oak_1 1.03-1.07 m
    from its trunk (a miss at 1 m), while whole-crown centroids hurt dense
    precision (0.85 -> 0.62). "auto" uses the crown centroid only for
    crowns that touch no neighbour."""
    chm, scanned = _live_sparse_map()
    truth = parse_tree_truth(str(WORLDS_DIR / "sparse_trees.sdf"))
    dets = detect_trees(chm, scanned, _shipped_params(position_mode="auto"),
                        -40.0, -40.0)
    metrics = score_detections(dets, truth, match_radius=1.0)
    assert (metrics["true_positives"], metrics["false_positives"]) == (5, 0)


def test_showcase_isolated_pine_is_detected():
    """Showcase demo run: pine_76 (nearest oak 7.0 m) was never baselined —
    heavy smoothing flattened its spire (raw 4.5 m -> 2.0 m). Two-scale
    treetops with the oak-height exclusion must find it, without adding
    false trees on the showcase map."""
    f = np.load(Path(__file__).parent / "data" / "showcase_live_canopy_map.npz")
    g = f["data"].astype(np.float32)
    scanned = g >= 0
    chm = np.where(scanned, g / 10.0, 0.0).astype(np.float32)
    dets = detect_trees(chm, scanned, _shipped_params(), -40.0, -40.0)
    assert min(math.hypot(d.x + 15.5, d.y + 23.0) for d in dets) <= 1.0
    truth = [t for t in parse_tree_truth(str(WORLDS_DIR / "showcase_forest.sdf"))
             if abs(t[1]) <= 30 and abs(t[2]) <= 30]
    inside = [d for d in dets if abs(d.x) <= 30 and abs(d.y) <= 30]
    m = score_detections(inside, truth, match_radius=1.0)
    assert m["precision"] >= 0.85 and m["recall"] >= 0.85


def test_narrow_peaks_off_keeps_old_behaviour():
    """narrow_sigma 0 disables the second scale (pine_76 missed again)."""
    f = np.load(Path(__file__).parent / "data" / "showcase_live_canopy_map.npz")
    g = f["data"].astype(np.float32)
    scanned = g >= 0
    chm = np.where(scanned, g / 10.0, 0.0).astype(np.float32)
    dets = detect_trees(chm, scanned, _shipped_params(narrow_sigma=0.0),
                        -40.0, -40.0)
    assert min(math.hypot(d.x + 15.5, d.y + 23.0) for d in dets) > 1.0


def _dense_map():
    f = np.load(Path(__file__).parent / "data" / "dense_live_canopy_map.npz")
    g = f["data"].astype(np.float32)
    scanned = g >= 0
    return np.where(scanned, g / 10.0, 0.0).astype(np.float32), scanned


def _nearest(dets, name, world="dense_forest"):
    t = {n: (x, y) for n, x, y in
         parse_tree_truth(str(WORLDS_DIR / f"{world}.sdf"))}[name]
    d = min(dets, key=lambda q: math.hypot(q.x - t[0], q.y - t[1]))
    return d, math.hypot(d.x - t[0], d.y - t[1])


OLD_SEEDS = dict(seed_lock=False, seed_snap=0.0, overlap_valley_ratio=0.0)


def test_seed_lock_keeps_neighbouring_oaks_apart():
    """Dense removal 8: oak_168's marker sat in a gap between branch clumps
    (raw), so oak_153's flood reached it first and the two oaks became one
    8.1 m crown — removing oak_153 left the crown standing on oak_168 and it
    was never LOST. Seed lock + snap gives oak_168 its own crown."""
    chm, sc = _dense_map()
    new = detect_trees(chm, sc, _shipped_params(), -40.0, -40.0)
    a, da = _nearest(new, "oak_153")
    b, db = _nearest(new, "oak_168")
    assert a is not b and da <= 2.0 and db <= 1.0
    old = detect_trees(chm, sc, _shipped_params(**OLD_SEEDS), -40.0, -40.0)
    assert _nearest(old, "oak_153")[0] is _nearest(old, "oak_168")[0]


def test_valley_keeps_a_separate_pine_out_of_an_oak_crown():
    """Dense removal 8: pine_75 (nearest oak trunk 5.8 m) had its own marker
    and a 2.4 m2 crown, but overlap suppression merged it into the oak's
    crown circle. A real valley between the two tops keeps it a tree."""
    chm, sc = _dense_map()
    new = detect_trees(chm, sc, _shipped_params(), -40.0, -40.0)
    assert _nearest(new, "pine_75")[1] <= 1.0
    old = detect_trees(chm, sc, _shipped_params(**OLD_SEEDS), -40.0, -40.0)
    assert _nearest(old, "pine_75")[1] > 1.5


def test_short_crowns_take_their_position_from_the_raw_spire():
    """Dense pine_163: heavy smoothing flattened its spire below the
    neighbouring crown edges, so the smoothed top band put it 1.9 m from its
    trunk (never baselined within 1.5 m). Short crowns use the raw band."""
    chm, sc = _dense_map()
    new = detect_trees(chm, sc, _shipped_params(), -40.0, -40.0)
    assert _nearest(new, "pine_163")[1] <= 1.0
    assert _nearest(new, "pine_111")[1] <= 0.5
    old = detect_trees(chm, sc, _shipped_params(band_surface="smoothed"),
                       -40.0, -40.0)
    assert _nearest(old, "pine_163")[1] > 1.5
    # broad oak crowns keep the smoothed band (unchanged position)
    assert abs(_nearest(new, "oak_168")[1] - _nearest(old, "oak_168")[1]) < 1e-6


def test_canopy_pines_do_not_hinge_on_the_long_range_saddle():
    """Dense run 8 freeze map (rebuilt from the raw scans): pine_75 and
    pine_163 each have a narrow peak and no tall oak within the exclusion
    radius, but the long-range valley test (to broad markers <= 8 m) sat at
    the height of the crown layer between pine and oak, so 2% map noise
    dropped both from the baseline. With it off they are found."""
    f = np.load(Path(__file__).parent / "data" / "dense_run8_freeze_map.npz")
    g = f["data"].astype(np.float32)
    scanned = g >= 0
    chm = np.where(scanned, g / 10.0, 0.0).astype(np.float32)
    new = detect_trees(chm, scanned, _shipped_params(), -40.0, -40.0)
    assert _nearest(new, "pine_75")[1] <= 1.0
    assert _nearest(new, "pine_163")[1] <= 1.0
    old = detect_trees(chm, scanned, _shipped_params(narrow_saddle_range=8.0),
                       -40.0, -40.0)
    assert _nearest(old, "pine_75")[1] > 1.5
