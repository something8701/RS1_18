"""Offline tests for the CHM tree detector.

These run without ROS or Gazebo, on synthetic canopy surfaces built from
known tree positions and on recorded live maps.
"""

import math
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
    """Shape filtering must drop 1-cell-wide rays without a hit-count
    threshold, while a compact crown seen only once is kept."""
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
    """A crown is not a circle: a long, lop-sided crown must stay one tree
    and keep its full size."""
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
    """Trees at least about 1.5 m apart must stay separate."""
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
    """When crowns overlap, the greedy overlap selection (Yang et al.)
    keeps one object instead of inventing extra treetops."""
    truth = scenario_layouts()[scenario]
    dets, metrics = run(truth)
    assert metrics["detected"] == 1
    assert metrics["false_positives"] == 0


def test_tight_quad_partially_splits_without_false_positives():
    """A 1.3 m quad has fully overlapping crowns. The watershed treats it as
    one canopy object and does not invent extra treetops."""
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
    # dense_forest spans about -37 to 37 m, so use the full 80 m grid.
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
    """Recorded sparse map 110: oak_5 splits into its crown and a lobe
    6.97 m from pine_2. If merging ran before overlap suppression, the lobe
    would absorb the pine (merge_radius 7.0). The lobe must be suppressed
    first and the pine kept."""
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
    """A broad 5 m crown over the trunk at (0, 0) plus a narrower, slightly
    taller lobe 2.5 m off-centre, which is the shape of a Fuel oak."""
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
    """A recorded live /forest_canopy_map of the sparse world, where oak_3's
    crown has an off-centre lobe."""
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
    """Real data: with lobe merging and crown positions every sparse tree is
    found within 0.5 m of its SDF trunk."""
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
    """With the older parameters, the top-band position put oak_3 about
    2.5 m from its trunk on this map (a miss at 1 m). The current parameters
    (smoothing 0.9) no longer split the lobe; see
    test_live_map_all_five_trees_within_half_a_metre."""
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
    """On sparse maps the top-band position put oak_1 1.03-1.07 m from its
    trunk (a miss at 1 m), while whole-crown centroids lower dense precision
    (0.85 to 0.62). "auto" uses the crown centroid only for crowns that touch
    no neighbour."""
    chm, scanned = _live_sparse_map()
    truth = parse_tree_truth(str(WORLDS_DIR / "sparse_trees.sdf"))
    dets = detect_trees(chm, scanned, _shipped_params(position_mode="auto"),
                        -40.0, -40.0)
    metrics = score_detections(dets, truth, match_radius=1.0)
    assert (metrics["true_positives"], metrics["false_positives"]) == (5, 0)


def test_showcase_isolated_pine_is_detected():
    """pine_76 (nearest oak 7.0 m) is flattened by heavy smoothing (raw
    4.5 m to 2.0 m). Two-scale treetops with the oak-height exclusion must
    find it without adding false trees on the showcase map."""
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
    """narrow_sigma 0 turns the second scale off, and pine_76 is missed."""
    f = np.load(Path(__file__).parent / "data" / "showcase_live_canopy_map.npz")
    g = f["data"].astype(np.float32)
    scanned = g >= 0
    chm = np.where(scanned, g / 10.0, 0.0).astype(np.float32)
    dets = detect_trees(chm, scanned, _shipped_params(narrow_sigma=0.0),
                        -40.0, -40.0)
    assert min(math.hypot(d.x + 15.5, d.y + 23.0) for d in dets) > 1.0
