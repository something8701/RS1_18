# Change detector and Nav2/SLAM (Anthony)

## Files in this update

| File | Status |
|---|---|
| `deforestation_monitoring/deforestation_monitoring/change_detector.py` | new |
| `deforestation_monitoring/test/test_change_core.py` | new (7 tests) |
| `deforestation_monitoring/deforestation_monitoring/scan_mapper.py` | modified |
| `deforestation_monitoring/launch/monitoring_nodes.launch.py` | modified |
| `deforestation_monitoring/launch/demo_full.launch.py` | modified |
| `deforestation_monitoring/setup.py` | modified (new `change_detector` entry point) |
| `41068_ignition_bringup/config/nav2_params_husky1.yaml` | modified |
| `41068_ignition_bringup/config/slam_params_husky1.yaml` | modified |

## Change detector (`change_detector.py`)

Compares the drone's canopy-height map (`/forest_canopy_map` from `scan_mapper`)
against a baseline and reports canopy loss.

- **Baseline**: accumulates per-cell max canopy height, then freezes when the drone
  completes its first lawnmower pass (`baseline_trigger: survey`; also `coverage`,
  `manual`). It refuses to freeze if less than `min_baseline_coverage` (30%) of the
  survey region was scanned (e.g. a broken lidar) and waits for the next pass.
  Saved to `/tmp/deforestation_eval/canopy_baseline.npz`.
- **Detection**: a cell is "lost" when it was canopy (>= 2 m) and dropped >= 1.5 m (measured crowns in `simple_trees` are only ~3-6 m tall), for
  3 consecutive updates. The drone lidar's scan lines are ~1.5 m apart (2 Hz at 3 m/s),
  so lost cells appear as stripes; these are joined with a morphological closing
  (`gap_bridge_m: 2.0`) before clustering, and a cluster must contain >= 8 *measured*
  lost cells. (The previous 3x3 opening erased every real detection because the
  stripes are 1 cell wide.)
- **Outputs**: `/canopy_change_map`, `/canopy_change_events`, `/canopy_change_markers`,
  `/drone_baseline_status`, `~/status`.
- **Services**: `~/snapshot_baseline`, `~/reset_baseline`.
- Core logic (`ChangeCore`) is numpy-only and unit-tested without ROS.

## `scan_mapper.py`

New parameter `enable_change_detection` (default `true`). The launch file sets it to
`false` so only `change_detector` publishes `/canopy_change_map`.

## Launch files

- `monitoring_nodes.launch.py`
  - Starts `change_detector` alongside `scan_mapper`.
  - World-aware tree removal: `simple_trees` removes `pine_9, oak_10, pine_11, pine_13`;
    `dense_forest` removes `pine_90, pine_91, pine_105, oak_106` (inside the drone patrol).
  - Static transform `husky1_map -> parrot1_odom` (x = 2.0 m, from the spawn poses),
    so Husky goals built from drone detections are no longer 2 m off.
  - `scan_mapper` swath cloud capped at 20,000 points; `drone_mapper` and
    `tree_fusion` log at WARN (both reduce CPU load).
- `demo_full.launch.py`: passes the `world` argument through.

## Nav2 (`nav2_params_husky1.yaml`)

- Global costmap fixed at 80 x 80 m (origin -40, -40), 0.2 m cells, obstacle +
  inflation layers. Previously it resized to the SLAM map and rejected goals outside it.
- Inflation radius 2.0 -> 1.0 m (2.0 closed the gaps between trees).
- Footprint from the Husky URDF: `[[0.52, 0.31], [0.52, -0.31], [-0.52, -0.31], [-0.52, 0.31]]`.
- `odom_topic: odom` (EKF output).
- `transform_tolerance: 1.0` on both costmaps. With the default 0.3 s, the SLAM
  transform (often ~0.5 s old in sim) failed, the goal was treated as (0, 0), and Nav2
  reported "Reached the goal!" without moving.
- Optional depth-camera obstacle source defined but disabled
  (`observation_sources: scan`; change to `scan depth` after checking alignment in RViz).

## SLAM (`slam_params_husky1.yaml`)

- `min_laser_range: 0.2` (matches the Husky lidar).
- `minimum_travel_distance: 0.3`, `minimum_travel_heading: 0.3` (stops graph nodes
  being added while stationary).

## Known issues in other nodes (not changed here)

- `tree_fusion`: fuses points from `parrot1_odom` and `husky1_map` without a TF
  transform, giving a 2 m offset between matching detections.
- `drone_mapper`: assumes a 15 m camera footprint; the 120 deg FOV at 10 m gives ~35 m.
- `simulate_tree_removal`: publishes one ground-truth point for all removed trees, so
  correct detections of other trees may score as false positives.
