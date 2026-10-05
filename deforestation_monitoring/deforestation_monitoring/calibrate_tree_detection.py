#!/usr/bin/env python3
"""Offline automatic calibration for the individual tree detector.

The Gazebo world SDF already stores the exact (x, y) position of every tree,
so manual labeling is unnecessary. This tool:

  1. reads that ground truth from a world file (or uses a named synthetic
     cluster scenario),
  2. builds the fused canopy height surface the detector will see,
  3. random-searches the detector parameters to maximise F1, and
  4. writes the best parameters as a ROS 2 parameter YAML plus a JSON report.

Examples:
    # Calibrate against the dense forest world (214 trees, many clusters):
    ros2 run deforestation_monitoring calibrate_tree_detection \
        --world /path/to/dense_forest.sdf \
        --out /tmp/tree_detection_params.yaml

    # Calibrate directly on a recorded drone terrain cloud (x,y,z CSV):
    ros2 run deforestation_monitoring calibrate_tree_detection \
        --world /path/to/world.sdf --cloud /tmp/terrain.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .tree_detection import (
    DetectionParams,
    calibrate,
    parse_tree_truth,
    params_to_ros_yaml,
    rasterize_points,
    save_calibration_json,
    scenario_layouts,
    synthetic_chm,
)


def _load_cloud(path: str) -> np.ndarray:
    suffix = Path(path).suffix.lower()
    if suffix == ".npy":
        data = np.load(path)
    else:  # csv / txt with x,y,z (and an optional first header row)
        data = np.genfromtxt(path, delimiter=",", skip_header=0)
        if data.ndim != 2 or data.shape[1] < 3:
            raise ValueError(
                f"{path} must contain at least x,y,z columns; "
                f"got shape {getattr(data, 'shape', None)}")
    return np.asarray(data[:, :3], dtype=np.float64)


def _build_calibration_input(args, truth):
    origin_x, origin_y = -40.0, -40.0
    dim_x = dim_y = int(80.0 / args.res)
    if args.cloud:
        points = _load_cloud(args.cloud)
        chm, hits = rasterize_points(
            points, args.res, origin_x, origin_y, dim_x, dim_y)
        scanned = hits > 0
        # Pad coverage with the overall map only where the cloud observed
        # something; there is no separate map in offline mode.
        return chm, scanned
    chm = synthetic_chm(
        truth, args.res, origin_x, origin_y, dim_x, dim_y,
        crown_factor=args.crown_factor, crown_offset=args.crown_offset,
        noise_sigma=args.noise_sigma, ground_relief=args.ground_relief,
        rng_seed=args.seed)
    scanned = np.ones(chm.shape, dtype=bool)
    return chm, scanned


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", default=None,
                        help="Gazebo world .sdf to read ground truth from")
    parser.add_argument("--scenario", default=None,
                        choices=sorted(scenario_layouts().keys()),
                        help="Named synthetic cluster scenario instead of a world")
    parser.add_argument("--cloud", default=None,
                        help="Optional recorded point cloud (.csv/.npy with x,y,z)")
    parser.add_argument("--out", default="/tmp/tree_detection_params.yaml",
                        help="Output ROS parameter YAML")
    parser.add_argument("--report", default="/tmp/tree_detection_calibration.json",
                        help="Output JSON calibration report")
    parser.add_argument("--res", type=float, default=0.2,
                        help="CHM resolution in metres")
    parser.add_argument("--trials", type=int, default=60,
                        help="Number of random parameter trials")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed for reproducible calibration")
    parser.add_argument("--match-radius", type=float, default=1.0,
                        help="Max trunk-to-treetop distance for a true positive (m)")
    parser.add_argument("--crown-factor", type=float, default=0.16,
                        help="Synthetic crown radius = factor*height + offset")
    parser.add_argument("--crown-offset", type=float, default=0.40,
                        help="Synthetic crown radius offset (m)")
    parser.add_argument("--noise-sigma", type=float, default=0.04,
                        help="Synthetic CHM noise (m)")
    parser.add_argument("--ground-relief", type=float, default=0.4,
                        help="Synthetic ground relief amplitude (m)")
    args = parser.parse_args(argv)

    if args.world and args.scenario:
        parser.error("choose either --world or --scenario, not both")
    if not args.world and not args.scenario:
        parser.error("one of --world or --scenario is required")

    if args.world:
        if not Path(args.world).is_file():
            parser.error(f"world file not found: {args.world}")
        truth = parse_tree_truth(args.world)
    else:
        truth = scenario_layouts()[args.scenario]

    if len(truth) < 2:
        print(f"Only {len(truth)} ground-truth trees found; nothing to calibrate.",
              file=sys.stderr)
        return 1

    chm, scanned = _build_calibration_input(args, truth)
    print(f"Calibrating on {len(truth)} ground-truth trees "
          f"({chm.shape[0]}x{chm.shape[1]} grid, {args.trials} trials)")
    result = calibrate(
        chm, scanned, truth,
        origin_x=-40.0, origin_y=-40.0,
        match_radius=args.match_radius,
        n_trials=args.trials,
        seed=args.seed,
    )

    best = result["best"]
    if best["params"] is None:
        print("Calibration produced no valid configuration.", file=sys.stderr)
        return 1

    metrics = best["metrics"]
    print("\nBest configuration:")
    for key, value in best["params"].items():
        print(f"  {key}: {value}")
    print("\nMetrics vs ground truth:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    params = DetectionParams.from_dict(best["params"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(params_to_ros_yaml(params))
    save_calibration_json(args.report, result)
    print(f"\nWrote ROS params to {out}")
    print(f"Wrote full report to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
