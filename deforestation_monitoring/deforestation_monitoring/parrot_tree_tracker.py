#!/usr/bin/env python3
"""Parrot Tree Tracker: finds individual trees from the drone's canopy
height model (CHM), freezes them as a baseline, then reports trees that go
LOST (cut) or GAINED.

The CHM is the canopy map (``/forest_canopy_map``), with cells it has not
scanned optionally filled from the depth-camera cloud (``/drone_terrain``).
Trees come from :mod:`tree_detection` (treetops on a height-dependent window,
crowns split by a saddle-aware watershed). Its parameters are in
``config/tree_detection_params.yaml``, calibrated against the world SDF by
``calibrate_tree_detection``.
"""

import collections
import dataclasses
import json
import math
import re

import numpy as np
import rclpy
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rcl_interfaces.msg import SetParametersResult
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header, String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Pose, Quaternion
from sensor_msgs.msg import PointCloud2, PointField

from deforestation_interfaces.msg import TreeChangeEvent, TreeStatus

from .survey_loops import LoopCounter
from .tree_detection import DetectedTree, DetectionParams, detect_trees
from .visibility import disc_max, top_drop


class ParrotTreeTracker(Node):
    """Detect individual trees from the fused CHM and track them over time."""

    def __init__(self):
        super().__init__('parrot_tree_tracker')

        # -- Topics / tracker parameters --
        self.declare_parameter('canopy_map_topic', '/forest_canopy_map')
        self.declare_parameter('change_map_topic', '/canopy_change_map')
        self.declare_parameter('baseline_status_topic', '/drone_baseline_status')
        self.declare_parameter('drone_terrain_topic', '/drone_terrain')
        self.declare_parameter('map_size_x', 80.0)
        self.declare_parameter('map_size_y', 80.0)
        self.declare_parameter('height_scale', 10.0)
        self.declare_parameter('min_baseline_cells', 150)
        self.declare_parameter('min_baseline_trees', 2)
        self.declare_parameter('baseline_coverage', 0.70)
        self.declare_parameter('baseline_stable_ticks', 3)
        # Baseline admission: a candidate seen in baseline_stable_ticks of the
        # last admit_window ticks (default 3 of 4), dropped after
        # candidate_max_misses misses in a row.
        self.declare_parameter('admit_window', 4)
        # > 0: admit on time instead (seen in admit_fraction of the ticks of
        # the last admit_window_s s), so live and replay tick rates agree.
        self.declare_parameter('admit_window_s', 0.0)
        self.declare_parameter('admit_fraction', 0.0)
        self.declare_parameter('candidate_max_misses', 2)
        self.declare_parameter('reset_candidates_at_plateau', True)
        self.declare_parameter('baseline_mean_position', False)
        # With baseline_mean_position: average over this many latest hits.
        self.declare_parameter('mean_position_hits', 12)
        # Before the freeze, detect on the per-cell time average of the CHM
        # instead of the latest map. The latest map reshapes crown edges on
        # every pass, so pines drop out and oak positions wander; on the
        # average they stay put. After the freeze the latest map is used, so
        # a cut shows at once.
        self.declare_parameter('baseline_mean_chm', False)
        # One felled oak can take a clump of its own crown (a separate
        # baseline "tree") with it. With merge_lost_crowns a LOST is held for
        # merge_window_s and dropped as part of the same crown if a larger oak
        # crown within merge_max_dist went LOST in that window. Pines are never
        # merged, so a pine cut under an oak is still reported. "Oak" = not
        # pine-coloured at the freeze, or without the camera a top of
        # >= merge_min_height.
        self.declare_parameter('merge_lost_crowns', False)
        self.declare_parameter('merge_window_s', 30.0)
        # A smaller crown that goes LOST after the oak is merged up to
        # merge_after_s later (often the next survey loop); no hold needed, so
        # nothing is delayed. Trade-off: a second oak felled within
        # merge_max_dist inside this time counts once.
        self.declare_parameter('merge_after_s', 600.0)
        # On that later side the host may be smaller, down to
        # merge_after_ratio x this crown's area: the baseline can split one
        # oak's crown into two halves of similar size (seen 0.73-0.93). Only a
        # host already reported counts, so two held crowns cannot merge into
        # each other. 0 = the host must be larger.
        self.declare_parameter('merge_after_ratio', 0.0)
        # A small oak crown (<= merge_hold_ratio x a standing oak crown within
        # merge_max_dist) is held up to merge_hold_s for that crown to go LOST
        # too: such clumps went LOST up to ~280 s before their oak. Clumps are
        # 0.09-0.17 of their neighbour, real oaks >= 0.40, so real oaks are
        # never held.
        self.declare_parameter('merge_hold_s', 0.0)
        self.declare_parameter('merge_hold_ratio', 0.25)
        self.declare_parameter('merge_max_dist', 5.5)
        self.declare_parameter('merge_min_height', 5.5)
        # Freeze on a coverage plateau: coverage (0-1) grew by less than
        # plateau_delta per tick for plateau_ticks ticks.
        self.declare_parameter('plateau_delta', 0.003)
        self.declare_parameter('plateau_ticks', 25)
        # Also wait for this many full survey loops (/survey_status); the
        # plateau alone comes at the end of the first. Same as scan_mapper
        # baseline_min_loops. 0 = plateau only.
        self.declare_parameter('freeze_min_loops', 0)
        # After the plateau, wait up to this many ticks for candidates still
        # being confirmed before freezing without them.
        self.declare_parameter('freeze_max_wait_ticks', 10)
        # Well under half the closest tree spacing (~1.4 m), so neighbours
        # never swap IDs.
        self.declare_parameter('track_radius', 0.6)
        self.declare_parameter('terrain_min_hits', 2)
        # Fill CHM cells the LiDAR has not scanned from the depth-camera cloud.
        # Its ground offset is only estimated, so the removal tests run
        # without it.
        self.declare_parameter('terrain_fill', True)
        self.declare_parameter('canopy_min_hits', 2)
        self.declare_parameter('min_gain_area_m2', 0.5)
        self.declare_parameter('gain_stable_ticks', 3)
        # A gain inside gain_crown_exclusion * radius of a baseline crown is
        # rejected as the same tree (1.0 = the crown's full extent).
        self.declare_parameter('gain_crown_exclusion', 1.0)
        self.declare_parameter('lost_streak_threshold', 3)
        # Loss cells (/canopy_change_map) needed near a tree for LOST. Cut oaks
        # show 13-150 within 2 m, standing trees <= 6.
        self.declare_parameter('lost_evidence_cells', 10)
        # GAINED needs the same kind of evidence: new-canopy cells near it.
        # A detection that only flickers back is not a gain.
        self.declare_parameter('gain_evidence_cells', 10)
        # Evidence search radius = crown radius, capped so a large crown
        # cannot collect change cells metres away.
        self.declare_parameter('evidence_max_radius', 2.0)
        # Also count height-drop cells (change map between -50 and 0: canopy
        # fell but not to ground) as LOST evidence. A tree cut from beside a
        # taller neighbour mostly falls onto the neighbour's crown, not to
        # ground.
        self.declare_parameter('lost_drop_evidence', False)
        # LOST also needs the tree's own top to be gone: the CHM max within
        # lost_top_radius of its baseline position dropped >= lost_top_drop
        # since the freeze. A cell count alone cannot tell the tree's crown
        # from a neighbour's or from speckle. 0.6 m = top noise of standing
        # trees. 0 = off.
        self.declare_parameter('lost_top_drop', 0.0)
        self.declare_parameter('lost_top_radius', 1.0)
        # And >= this fraction of the cells that formed its top at the freeze
        # fell >= 0.6 m (visibility.top_drop 'frac'): one neighbour cell can
        # hold up the max, but not the fraction. 0 = off.
        self.declare_parameter('lost_top_frac', 0.0)
        # Or: the disc mean fell >= lost_mean_drop m with >= lost_mean_frac of
        # the top cells down. One branch tip of a neighbour can keep the max
        # of a felled oak; the mean still drops ~3 m. No standing tree in 15
        # dense recordings ever lost 2 m of mean. 0 = off.
        self.declare_parameter('lost_mean_drop', 0.0)
        self.declare_parameter('lost_mean_frac', 0.6)
        # Camera LOST evidence, for pines cut from under oak crowns (no ground
        # exposed, branch tips keep the LiDAR top). A pine-coloured tree (blue/
        # green >= seen_pine) whose spot read >= seen_canopy at the freeze has
        # evidence once its last two camera visits (lane passes, split by gaps
        # > seen_visit_gap_s) each averaged < seen_open canopy, and its top fell
        # >= lost_top_drop. Whole visits, because single oblique readings at a
        # crown edge dip low. False = off.
        self.declare_parameter('seen_lost', False)
        self.declare_parameter('seen_canopy', 0.6)
        self.declare_parameter('seen_open', 0.3)
        self.declare_parameter('seen_radius', 0.75)
        self.declare_parameter('seen_min_cells', 4)
        self.declare_parameter('seen_visit_gap_s', 15.0)
        self.declare_parameter('seen_pine', 0.65)
        # Pines the baseline does not have (inside an oak's crown, or missing
        # at the freeze): pine-coloured canopy patches of >= area_min_cells
        # with no baseline tree within area_exclude m are watched with the
        # same camera rule; one that fires publishes a CANOPY LOST alert on
        # /canopy_change_events. False = off.
        self.declare_parameter('camera_area_alerts', False)
        self.declare_parameter('area_min_cells', 6)
        self.declare_parameter('area_exclude', 1.5)
        # Disc (m) for the camera rule and the drop at a patch. The patch
        # centre is the pine's own crown; neighbours' crowns reach into a
        # larger disc around a cut pine.
        self.declare_parameter('area_radius', 0.5)
        # A patch whose nearby baseline trees are all non-pine is a pine the
        # baseline placed off its crown: the patch becomes the nearest tree's
        # camera site, so that tree can go LOST on the camera.
        self.declare_parameter('area_attach', False)
        # At a patch the loss is checked on the disc mean (m), not the max: a
        # neighbour's branch often holds the max of a cut pine. 0 = use
        # lost_top_drop on the max.
        self.declare_parameter('area_mean_drop', 1.0)
        # A pine patch of >= area_admit_cells where the LiDAR saw a tree
        # within area_admit_dist m in any of the last area_admit_ticks ticks
        # before the freeze joins the baseline, watched at its patch. Pines
        # under several oak crowns flicker on the map and miss the baseline
        # otherwise. Dense pine patches are 7-33 cells; oak-edge patches in
        # the sparse world 6-8. 0 = off.
        self.declare_parameter('area_admit_cells', 12)
        self.declare_parameter('area_admit_ticks', 60)
        self.declare_parameter('area_admit_dist', 1.0)
        # A baseline tree enters at the median of its detections over the
        # last freeze_median_ticks before the freeze (nearest one per tick
        # within freeze_median_radius m), not its latest one: the last lanes
        # can shift a dense oak's top 2 m for a few ticks. With
        # freeze_median_exclusive, detections shared by two freeze-tick trees
        # are skipped (see _median_position). 0 = the latest detection.
        self.declare_parameter('freeze_median_ticks', 0)
        self.declare_parameter('freeze_median_radius', 1.5)
        self.declare_parameter('freeze_median_exclusive', False)
        # A pine patch of >= area_camera_cells with no baseline tree within
        # area_camera_clear m joins the baseline as a camera-only tree, even
        # without a LiDAR hit. It is left out of LiDAR matching, so only the
        # camera rule can report it LOST. 0 = off.
        self.declare_parameter('area_camera_cells', 20)
        # Pine patches (alerts, sites, admission) only this far inside the
        # survey box: near the edge of the scanned area the camera sees the
        # canopy only obliquely.
        self.declare_parameter('area_edge_margin', 0.0)
        self.declare_parameter('area_camera_clear', 2.0)
        # Pine tips by camera colour: a CHM local maximum (window
        # colour_spire_window m, top >= colour_spire_min_top m) that reads
        # >= colour_spire_bg within 0.5 m and has no detection within
        # colour_spire_merge m is added as a tree, until the freeze. Off: at
        # oak crown edges it adds false trees (they read pine-coloured there).
        self.declare_parameter('colour_pines', False)
        self.declare_parameter('colour_spire_bg', 0.65)
        self.declare_parameter('colour_spire_window', 1.25)
        self.declare_parameter('colour_spire_min_top', 3.0)
        self.declare_parameter('colour_spire_merge', 1.5)
        self.declare_parameter('survey_x_min', -40.0)
        self.declare_parameter('survey_x_max', 40.0)
        self.declare_parameter('survey_y_min', -40.0)
        self.declare_parameter('survey_y_max', 40.0)
        self.declare_parameter('publish_rate', 1.0)

        # -- Individual tree detector parameters (from the calibrated YAML) --
        defaults = DetectionParams()
        for key, value in defaults.to_dict().items():
            self.declare_parameter(key, value)

        self.canopy_topic = self.get_parameter('canopy_map_topic').value
        self.change_topic = self.get_parameter('change_map_topic').value
        self.baseline_topic = self.get_parameter('baseline_status_topic').value
        self.terrain_topic = self.get_parameter('drone_terrain_topic').value
        self.map_x = float(self.get_parameter('map_size_x').value)
        self.map_y = float(self.get_parameter('map_size_y').value)
        self.height_scale = float(self.get_parameter('height_scale').value)
        self.min_baseline_cells = int(self.get_parameter('min_baseline_cells').value)
        self.min_baseline_trees = int(self.get_parameter('min_baseline_trees').value)
        self.baseline_coverage = float(self.get_parameter('baseline_coverage').value)
        self.baseline_stable_ticks = int(
            self.get_parameter('baseline_stable_ticks').value)
        self.admit_window = int(self.get_parameter('admit_window').value)
        self.admit_window_s = float(self.get_parameter('admit_window_s').value)
        self.admit_fraction = float(self.get_parameter('admit_fraction').value)
        self.candidate_max_misses = int(
            self.get_parameter('candidate_max_misses').value)
        self.baseline_mean_chm = bool(self.get_parameter('baseline_mean_chm').value)
        self.merge_lost_crowns = bool(self.get_parameter('merge_lost_crowns').value)
        self.merge_window_s = float(self.get_parameter('merge_window_s').value)
        self.merge_after_s = float(self.get_parameter('merge_after_s').value)
        self.merge_after_ratio = float(self.get_parameter('merge_after_ratio').value)
        self.merge_hold_s = float(self.get_parameter('merge_hold_s').value)
        self.merge_hold_ratio = float(self.get_parameter('merge_hold_ratio').value)
        self.merge_max_dist = float(self.get_parameter('merge_max_dist').value)
        self.merge_min_height = float(self.get_parameter('merge_min_height').value)
        self._pending_lost = []     # (time, tree) held for merging
        self._reported_lost = []    # (time, tree) published
        self._chm_sum = None        # running CHM sum / sample count until the freeze
        self._chm_n = None
        self.reset_candidates_at_plateau = bool(
            self.get_parameter('reset_candidates_at_plateau').value)
        self.baseline_mean_position = bool(
            self.get_parameter('baseline_mean_position').value)
        self.mean_position_hits = max(
            1, int(self.get_parameter('mean_position_hits').value))
        self.plateau_delta = float(self.get_parameter('plateau_delta').value)
        self.plateau_ticks = int(self.get_parameter('plateau_ticks').value)
        self.freeze_min_loops = int(self.get_parameter('freeze_min_loops').value)
        self.loop_counter = LoopCounter()
        self.freeze_max_wait_ticks = int(
            self.get_parameter('freeze_max_wait_ticks').value)
        self.track_radius = float(self.get_parameter('track_radius').value)
        self.terrain_min_hits = int(self.get_parameter('terrain_min_hits').value)
        self.terrain_fill = bool(self.get_parameter('terrain_fill').value)
        self.canopy_min_hits = int(self.get_parameter('canopy_min_hits').value)
        self.min_gain_area_m2 = float(self.get_parameter('min_gain_area_m2').value)
        self.gain_stable_ticks = int(self.get_parameter('gain_stable_ticks').value)
        self.gain_crown_exclusion = float(
            self.get_parameter('gain_crown_exclusion').value)
        self.lost_streak_threshold = int(
            self.get_parameter('lost_streak_threshold').value)
        self.lost_evidence_cells = int(
            self.get_parameter('lost_evidence_cells').value)
        self.gain_evidence_cells = int(
            self.get_parameter('gain_evidence_cells').value)
        self.evidence_max_radius = float(
            self.get_parameter('evidence_max_radius').value)
        self.lost_drop_evidence = bool(
            self.get_parameter('lost_drop_evidence').value)
        self.lost_top_drop = float(self.get_parameter('lost_top_drop').value)
        self.lost_top_radius = float(
            self.get_parameter('lost_top_radius').value)
        self.lost_top_frac = float(self.get_parameter('lost_top_frac').value)
        self.lost_mean_drop = float(self.get_parameter('lost_mean_drop').value)
        self.lost_mean_frac = float(self.get_parameter('lost_mean_frac').value)
        self.seen_lost = bool(self.get_parameter('seen_lost').value)
        self.seen_canopy = float(self.get_parameter('seen_canopy').value)
        self.seen_open = float(self.get_parameter('seen_open').value)
        self.seen_radius = float(self.get_parameter('seen_radius').value)
        self.seen_min_cells = int(self.get_parameter('seen_min_cells').value)
        self.seen_visit_gap_s = float(self.get_parameter('seen_visit_gap_s').value)
        self.seen_pine = float(self.get_parameter('seen_pine').value)
        self.camera_area_alerts = bool(self.get_parameter('camera_area_alerts').value)
        self.area_min_cells = int(self.get_parameter('area_min_cells').value)
        self.area_exclude = float(self.get_parameter('area_exclude').value)
        self.area_radius = float(self.get_parameter('area_radius').value)
        self.area_attach = bool(self.get_parameter('area_attach').value)
        self.area_mean_drop = float(self.get_parameter('area_mean_drop').value)
        self.area_admit_cells = int(self.get_parameter('area_admit_cells').value)
        self.area_admit_dist = float(self.get_parameter('area_admit_dist').value)
        self.area_camera_cells = int(self.get_parameter('area_camera_cells').value)
        self.area_edge_margin = float(self.get_parameter('area_edge_margin').value)
        self.freeze_median_ticks = int(self.get_parameter('freeze_median_ticks').value)
        self.freeze_median_radius = float(self.get_parameter('freeze_median_radius').value)
        self.freeze_median_exclusive = bool(self.get_parameter('freeze_median_exclusive').value)
        self.area_camera_clear = float(self.get_parameter('area_camera_clear').value)
        self._recent_dets = collections.deque(
            maxlen=max(1, int(self.get_parameter('area_admit_ticks').value),
                       int(self.get_parameter('freeze_median_ticks').value)))
        self._pine_patches = []     # watched pine-coloured patches without a baseline tree
        self._alerted_patches = set()
        self.colour_pines = bool(self.get_parameter('colour_pines').value)
        self.colour_spire_bg = float(self.get_parameter('colour_spire_bg').value)
        self.colour_spire_window = float(self.get_parameter('colour_spire_window').value)
        self.colour_spire_min_top = float(self.get_parameter('colour_spire_min_top').value)
        self.colour_spire_merge = float(self.get_parameter('colour_spire_merge').value)
        self.survey_x_min = float(self.get_parameter('survey_x_min').value)
        self.survey_x_max = float(self.get_parameter('survey_x_max').value)
        self.survey_y_min = float(self.get_parameter('survey_y_min').value)
        self.survey_y_max = float(self.get_parameter('survey_y_max').value)

        params_dict = defaults.to_dict()
        for key in params_dict:
            params_dict[key] = self.get_parameter(key).value
        self.itd_params = DetectionParams.from_dict(params_dict)
        self.add_on_set_parameters_callback(self._on_params_changed)

        self.res = float(self.itd_params.chm_resolution)
        self.dim_x = int(self.map_x / self.res)
        self.dim_y = int(self.map_y / self.res)
        self.origin_x = -self.map_x / 2.0
        self.origin_y = -self.map_y / 2.0

        # Accumulators
        self.terrain_height = np.zeros((self.dim_x, self.dim_y), dtype=np.float32)
        self.terrain_hits = np.zeros((self.dim_x, self.dim_y), dtype=np.int32)
        self.terrain_points = 0

        self.canopy_data = None     # [height, width] raw map values, -1 unscanned
        self.canopy_hits = None     # [height, width] per-cell observation count
        self.canopy_res = None
        self.canopy_ox = None
        self.canopy_oy = None

        self.change_lost = None     # bool grid of lost cells from the change map
        self.change_dropped = None  # height-drop cells (not to ground)
        self.change_gained = None   # bool grid of new-canopy cells
        self.change_res = None
        self.change_ox = None
        self.change_oy = None

        # Camera grids from camera_species_mapper ([iy, ix], NaN = no data):
        # colour (blue/green), canopy fraction so far, recent canopy fraction.
        # Per baseline tree, the camera visit in progress and the means of
        # the last two finished visits.
        self.species_mean = None
        self.seen_map = None
        self.seen_recent = None
        self.camera_grid = None     # (res, origin_x, origin_y)
        self._seen_visits = {}

        # Tracking state
        self.baseline_trees = []    # [{id, x, y, height, radius_m, area_m2}]
        self.baseline_frozen = False
        self.baseline_tree_count = 0
        self.baseline_scanned = None
        self.baseline_chm = None    # CHM at freeze, -1 = unscanned
        self._current_chm = None    # this tick's CHM, -1 = unscanned
        self.drone_baseline_ready = False
        self.coverage_pct = 0.0
        self.next_tree_id = 1
        self.total_trees_lost = 0
        self.total_trees_gained = 0
        self._alerted_lost_ids = set()
        self._unmatched_streaks = {}
        self._prev_dets = []
        self._stable_count = 0
        self._last_det_count = -1
        self._tick = 0
        self._pre_freeze_ticks = 0
        self._gains_skipped_unobserved = 0
        self._gains_rejected_in_crown = 0
        self._gains_rejected_no_evidence = 0
        self._pending_gains = []
        self._freeze_candidates = []
        self._plateau_reached = False
        self._ticks_since_plateau = 0
        self._plateau_time = None
        self._disp_pre = []
        self._disp_post = []
        self._last_cov = 0.0
        self._cov_stable_ticks = 0

        # -- QoS --
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        default_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # -- Subscribers --
        self.canopy_sub = self.create_subscription(
            OccupancyGrid, self.canopy_topic, self.canopy_cb, map_qos)
        self.hits_sub = self.create_subscription(
            OccupancyGrid, '/forest_canopy_hits', self.hits_cb, map_qos)
        self.change_sub = self.create_subscription(
            OccupancyGrid, self.change_topic, self.change_cb, map_qos)
        self.baseline_sub = self.create_subscription(
            String, self.baseline_topic, self.baseline_cb, default_qos)
        self.coverage_sub = self.create_subscription(
            String, '/scan_coverage', self.coverage_cb, default_qos)
        if self.freeze_min_loops > 0:
            self.create_subscription(String, '/survey_status', self.survey_cb, default_qos)
        self.terrain_sub = self.create_subscription(
            PointCloud2, self.terrain_topic, self.terrain_cb, sensor_qos)
        self.create_subscription(
            OccupancyGrid, '/forest_species_map', self.species_map_cb, map_qos)
        self.create_subscription(
            OccupancyGrid, '/forest_canopy_seen_map', self.seen_map_cb, map_qos)
        self.create_subscription(
            OccupancyGrid, '/forest_canopy_seen', self.seen_recent_cb, map_qos)

        # -- Publishers --
        self.status_pub = self.create_publisher(
            TreeStatus, '/parrot_tree_status', default_qos)
        self.event_pub = self.create_publisher(
            TreeChangeEvent, '/parrot_tree_change_events', default_qos)
        # Camera area alerts share scan_mapper's alert topic and format.
        self.canopy_alert_pub = self.create_publisher(
            String, '/canopy_change_events', default_qos)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/parrot_tree_change_markers', default_qos)
        self.positions_pub = self.create_publisher(
            PointCloud2, '/parrot_tree_positions', default_qos)
        self.treetop_pub = self.create_publisher(
            MarkerArray, '/parrot_treetop_markers', default_qos)
        self.crown_pub = self.create_publisher(
            OccupancyGrid, '/parrot_crown_map', map_qos)
        # Frozen baseline (x, y, z = tree id), latched, for tests/dashboards.
        self.baseline_pub = self.create_publisher(
            PointCloud2, '/parrot_tree_baseline', map_qos)
        # The same baseline as JSON with what the dashboard shows per tree
        # (source, species colour) and the watched pine patches; latched.
        self.baseline_info_pub = self.create_publisher(
            String, '/parrot_tree_baseline_info', map_qos)
        # One JSON note per LOST (what caught it: height map or camera) and
        # per merged crown, for the dashboard.
        self.change_note_pub = self.create_publisher(
            String, '/parrot_tree_change_notes', default_qos)
        self.drone_baseline_pub = self.create_publisher(
            String, '/drone_baseline_status', default_qos)

        self.reset_srv = self.create_service(
            Trigger, '~/reset_baseline', self.reset_baseline_callback)

        period = 1.0 / max(0.5, float(self.get_parameter('publish_rate').value))
        self.create_timer(period, self.publish_status)

        self.get_logger().info(
            f'Parrot Tree Tracker ready (CHM ITD). Grid: {self.dim_x}x{self.dim_y} '
            f'at {self.res}m. window_scale={self.itd_params.window_scale}, '
            f'saddle_ratio={self.itd_params.saddle_ratio}, '
            f'smooth_sigma={self.itd_params.smooth_sigma}.'
        )

    def _on_params_changed(self, params):
        """Apply live parameter changes so the detector can be recalibrated
        in place without restarting the node."""
        fields = set(self.itd_params.to_dict().keys())
        for param in params:
            if param.name == 'chm_resolution':
                continue  # the accumulator grid size is fixed at startup
            if param.name == 'baseline_coverage':
                self.baseline_coverage = float(param.value)
                continue
            if param.name == 'baseline_stable_ticks':
                self.baseline_stable_ticks = int(param.value)
                continue
            if param.name == 'plateau_delta':
                self.plateau_delta = float(param.value)
                continue
            if param.name == 'plateau_ticks':
                self.plateau_ticks = int(param.value)
                continue
            if param.name in fields:
                setattr(self.itd_params, param.name, param.value)
        self.get_logger().info(
            f'ITD params updated: smooth={self.itd_params.smooth_sigma}, '
            f'min_height={self.itd_params.min_height}, '
            f'window_scale={self.itd_params.window_scale}, '
            f'saddle={self.itd_params.saddle_ratio}, '
            f'min_crown_cells={self.itd_params.min_crown_cells}')
        return SetParametersResult(successful=True)

    # ── Callbacks ────────────────────────────────────────────────────

    def terrain_cb(self, cloud: PointCloud2):
        """Accumulate the drone depth cloud as a max-height grid."""
        offsets = {}
        for field in cloud.fields:
            if field.name in ('x', 'y', 'z'):
                offsets[field.name] = field.offset
        if len(offsets) != 3 or cloud.point_step <= 0 or not cloud.data:
            return

        raw = np.frombuffer(bytes(cloud.data), dtype=np.uint8)
        usable = raw.size - (raw.size % cloud.point_step)
        if usable <= 0:
            return
        arr = raw[:usable].reshape(-1, cloud.point_step)
        dtype = '>f4' if cloud.is_bigendian else '<f4'

        def col(name):
            off = offsets[name]
            return arr[:, off:off + 4].copy().view(dtype).ravel()

        xs, ys, zs = col('x'), col('y'), col('z')
        valid = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
        xs, ys, zs = xs[valid], ys[valid], zs[valid]
        if len(xs) == 0:
            return

        ix = ((xs - self.origin_x) / self.res).astype(np.int64)
        iy = ((ys - self.origin_y) / self.res).astype(np.int64)
        inside = (ix >= 0) & (ix < self.dim_x) & (iy >= 0) & (iy < self.dim_y)
        ix, iy, zs = ix[inside], iy[inside], zs[inside]
        if len(ix) == 0:
            return
        np.maximum.at(self.terrain_height, (ix, iy), zs)
        np.add.at(self.terrain_hits, (ix, iy), 1)
        self.terrain_points += len(ix)

    def canopy_cb(self, grid: OccupancyGrid):
        data = np.array(grid.data, dtype=np.float32).reshape(
            grid.info.height, grid.info.width)
        self.canopy_data = data
        self.canopy_res = grid.info.resolution
        self.canopy_ox = grid.info.origin.position.x
        self.canopy_oy = grid.info.origin.position.y

    def hits_cb(self, grid: OccupancyGrid):
        self.canopy_hits = np.array(grid.data, dtype=np.int16).reshape(
            grid.info.height, grid.info.width)

    def change_cb(self, grid: OccupancyGrid):
        data = np.array(grid.data, dtype=np.int8).reshape(
            grid.info.height, grid.info.width)
        self.change_lost = data <= -50
        self.change_dropped = (data < 0) & (data > -50)
        self.change_gained = data >= 50
        self.change_res = grid.info.resolution
        self.change_ox = grid.info.origin.position.x
        self.change_oy = grid.info.origin.position.y

    def _camera_grid(self, grid: OccupancyGrid):
        """A camera_species_mapper grid as [iy, ix] floats (value / 100, NaN = no data)."""
        self.camera_grid = (grid.info.resolution, grid.info.origin.position.x,
                            grid.info.origin.position.y)
        data = np.array(grid.data, dtype=np.float32).reshape(
            grid.info.height, grid.info.width)
        return np.where(data >= 0, data / 100.0, np.nan)

    def species_map_cb(self, grid: OccupancyGrid):
        self.species_mean = self._camera_grid(grid)

    def seen_map_cb(self, grid: OccupancyGrid):
        self.seen_map = self._camera_grid(grid)

    def seen_recent_cb(self, grid: OccupancyGrid):
        self.seen_recent = self._camera_grid(grid)

    def _disc_mean(self, grid, tree, radius=None):
        """Mean of a camera grid within `radius` (default seen_radius) of a
        tree; None if fewer than seen_min_cells cells have data."""
        if grid is None or self.camera_grid is None:
            return None
        radius = self.seen_radius if radius is None else radius
        res, ox, oy = self.camera_grid
        k = int(math.ceil(radius / res))
        ix, iy = int((tree['x'] - ox) / res), int((tree['y'] - oy) / res)
        x0, x1 = max(0, ix - k), min(grid.shape[1], ix + k + 1)
        y0, y1 = max(0, iy - k), min(grid.shape[0], iy + k + 1)
        if x0 >= x1 or y0 >= y1:
            return None
        yy, xx = np.mgrid[y0:y1, x0:x1]
        disc = (xx - ix) ** 2 + (yy - iy) ** 2 <= (radius / res) ** 2
        vals = grid[y0:y1, x0:x1][disc]
        vals = vals[np.isfinite(vals)]
        return float(vals.mean()) if vals.size >= self.seen_min_cells else None

    def _find_pine_patches(self):
        """Pine-coloured canopy patches with no baseline tree nearby (at the
        freeze). With area_attach, a patch whose nearby trees are none of them
        camera pines becomes the nearest one's camera site (tree['site'])."""
        if self.species_mean is None or self.seen_map is None:
            return []
        res, ox, oy = self.camera_grid
        mask = ((np.nan_to_num(self.species_mean) >= self.seen_pine)
                & (np.nan_to_num(self.seen_map) >= self.seen_canopy))      # [iy, ix]
        labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
        if n == 0:
            return []
        index = range(1, n + 1)
        sizes = ndimage.sum(mask, labels, index)
        centres = ndimage.center_of_mass(mask, labels, index)
        patches = []
        for k, cells, (row, col) in zip(index, sizes, centres):
            x, y = ox + (col + 0.5) * res, oy + (row + 0.5) * res
            m = self.area_edge_margin
            if (cells < self.area_min_cells
                    or not (self.survey_x_min + m <= x <= self.survey_x_max - m
                            and self.survey_y_min + m <= y <= self.survey_y_max - m)):
                continue
            patch = {'id': f'P{k}', 'x': x, 'y': y, 'cells': int(cells), 'r': self.area_radius}
            near = [(math.hypot(x - t['x'], y - t['y']), t['id'], t) for t in self.baseline_trees
                    if math.hypot(x - t['x'], y - t['y']) <= self.area_exclude]
            if not near:
                patches.append(patch)
            elif self.area_attach and not any(t.get('camera_pine') for _, _, t in near):
                host = min(near)[2]
                if host.get('site') is None or host['site']['cells'] < patch['cells']:
                    host['site'] = patch
                    host['colour'] = None      # its own disc reads a neighbour's colour
        return patches

    def _median_position(self, tree, final=()):
        """Median of the tree's detections over the last freeze_median_ticks
        pre-freeze ticks (nearest one per tick within freeze_median_radius);
        its own position if it was seen in fewer than 3.

        With freeze_median_exclusive, a past detection within the radius of
        two or more freeze-tick trees (``final``) is skipped: it may be those
        crowns fused into one, and using it would pull both trees onto it."""
        rad = self.freeze_median_radius
        xs, ys = [], []
        for tick in list(self._recent_dets)[-self.freeze_median_ticks:]:
            near = [d for d in tick if math.hypot(d.x - tree['x'], d.y - tree['y']) <= rad]
            if self.freeze_median_exclusive and final:
                near = [d for d in near
                        if sum(math.hypot(d.x - fx, d.y - fy) <= rad for fx, fy in final) <= 1]
            if near:
                d = min(near, key=lambda d: math.hypot(d.x - tree['x'], d.y - tree['y']))
                xs.append(d.x)
                ys.append(d.y)
        if len(xs) < 3:
            return tree['x'], tree['y']
        return float(np.median(xs)), float(np.median(ys))

    def _admit_pine_patches(self):
        """Move watched pine patches into the baseline: those the LiDAR also
        saw a tree at (recently, before the freeze), then large ones clear of
        every baseline tree as camera-only trees. Returns the new trees."""
        if not self.camera_area_alerts:
            return []
        admitted = []
        for p in list(self._pine_patches):
            if self.area_admit_cells <= 0 or p['cells'] < self.area_admit_cells:
                continue
            hits = [[d for d in tick
                     if math.hypot(d.x - p['x'], d.y - p['y']) <= self.area_admit_dist]
                    for tick in self._recent_dets]
            hits = [h for h in hits if h]
            if not hits:
                continue
            d = min(hits[-1], key=lambda d: math.hypot(d.x - p['x'], d.y - p['y']))
            if any(math.hypot(d.x - t['x'], d.y - t['y']) <= self.area_exclude
                   for t in admitted):
                continue
            tree = {'id': len(self.baseline_trees) + 1, 'x': d.x, 'y': d.y,
                    'height': d.height, 'radius_m': d.radius_m, 'area_m2': d.area_m2,
                    'colour': self._disc_mean(self.species_mean, p, p['r']),
                    'camera_pine': True, 'site': p, 'hits': len(hits)}
            self.baseline_trees.append(tree)
            self._pine_patches.remove(p)
            admitted.append(tree)
        if self.area_camera_cells > 0:
            for p in list(self._pine_patches):
                if p['cells'] < self.area_camera_cells or any(
                        math.hypot(p['x'] - t['x'], p['y'] - t['y']) <= self.area_camera_clear
                        for t in self.baseline_trees):
                    continue
                area = p['cells'] * self.camera_grid[0] ** 2
                tree = {'id': len(self.baseline_trees) + 1, 'x': p['x'], 'y': p['y'],
                        'height': 0.0, 'radius_m': max(1.0, math.sqrt(area / math.pi)),
                        'area_m2': area,
                        'colour': self._disc_mean(self.species_mean, p, p['r']),
                        'camera_pine': True, 'camera_only': True, 'site': p, 'hits': 0}
                self.baseline_trees.append(tree)
                self._pine_patches.remove(p)
                admitted.append(tree)
        return admitted

    def _check_pine_patches(self):
        """Publish a CANOPY LOST alert for each watched patch the camera now sees through."""
        for p in self._pine_patches:
            if p['id'] in self._alerted_patches or not self._seen_through(p):
                continue
            if not self._site_dropped(p):
                continue
            self._alerted_patches.add(p['id'])
            text = (f"CANOPY LOST: camera sees through pine canopy (~{p['cells']} cells) "
                    f"near ({p['x']:.1f}, {p['y']:.1f})")
            self.canopy_alert_pub.publish(String(data=text))
            self.get_logger().info(text)

    def _update_seen(self, now):
        """Average the canopy fraction per camera visit, for each baseline pine
        (and each watched pine patch)."""
        sites = [t for t in self.baseline_trees if t.get('camera_pine') or t.get('site')]
        if self.camera_area_alerts:
            sites += self._pine_patches
        for tree in sites:
            site = tree.get('site') or tree
            v = self._seen_visits.setdefault(
                tree['id'], {'t': None, 'sum': 0.0, 'n': 0, 'done': []})
            if v['n'] and now - v['t'] > self.seen_visit_gap_s:
                mean = v['sum'] / v['n']
                v['done'] = (v['done'] + [mean])[-2:]
                v['sum'], v['n'] = 0.0, 0
                if mean < self.seen_open:
                    self.get_logger().info(
                        f'camera saw through #{tree["id"]} ({site["x"]:.1f}, '
                        f'{site["y"]:.1f}): visit mean {mean:.2f}')
            seen = self._disc_mean(self.seen_recent, site, site.get('r'))
            if seen is not None:
                v['sum'] += seen
                v['n'] += 1
                v['t'] = now

    def _colour_spires(self, chm, scanned, detections):
        """Pine tips the height detector missed, from the CHM and camera colour."""
        if self.species_mean is None:
            return []
        k = max(3, int(round(self.colour_spire_window / self.res)) | 1)
        top = np.where(scanned, chm, 0.0)
        peak = (top == ndimage.maximum_filter(top, size=k)) & (top >= self.colour_spire_min_top)
        ix, iy = np.nonzero(peak)
        order = np.argsort(-top[ix, iy])              # tallest first
        taken = [(d.x, d.y) for d in detections]
        added = []
        for i in order:
            x = self.origin_x + (ix[i] + 0.5) * self.res
            y = self.origin_y + (iy[i] + 0.5) * self.res
            if not (self.survey_x_min <= x <= self.survey_x_max
                    and self.survey_y_min <= y <= self.survey_y_max):
                continue
            if any(math.hypot(x - tx, y - ty) < self.colour_spire_merge for tx, ty in taken):
                continue
            bg = self._disc_mean(self.species_mean, {'x': x, 'y': y}, radius=0.5)
            if bg is None or bg < self.colour_spire_bg:
                continue
            taken.append((x, y))
            added.append(DetectedTree(id=0, x=x, y=y, height=float(top[ix[i], iy[i]]),
                                      area_m2=math.pi, radius_m=1.0,
                                      peak_ix=int(ix[i]), peak_iy=int(iy[i])))
        return added

    def _top_fell(self, tree) -> bool:
        """The height part of the LOST rule: the disc max fell >= lost_top_drop
        with >= lost_top_frac of the top cells down, or the disc mean fell
        >= lost_mean_drop with >= lost_mean_frac of them down."""
        frac = self._top_drop(tree, 'frac') if (self.lost_top_frac > 0
                                                 or self.lost_mean_drop > 0) else 1.0
        top = ((self.lost_top_drop <= 0   # heights are 0.1 m steps
                or self._top_drop(tree) >= self.lost_top_drop - 1e-6)
               and (self.lost_top_frac <= 0 or frac >= self.lost_top_frac - 1e-6))
        if top:
            return True
        return (self.lost_mean_drop > 0
                and frac >= self.lost_mean_frac - 1e-6
                and self._top_drop(tree, 'mean') >= self.lost_mean_drop - 1e-6)

    def _site_dropped(self, site) -> bool:
        """The height map agrees with the camera: a patch's disc mean fell
        >= area_mean_drop (area_radius); a baseline pine's top fell
        >= lost_top_drop (disc max, seen_radius) or its disc mean fell
        >= area_mean_drop. The mean matters because a neighbour's branch tip
        can hold the max for minutes after the pine is gone."""
        if 'r' in site and self.area_mean_drop > 0:
            return (self._top_drop(site, mode='mean', radius=site['r'])
                    >= self.area_mean_drop - 1e-6)
        if (self.area_mean_drop > 0 and self._top_drop(site, mode='mean', radius=self.seen_radius)
                >= self.area_mean_drop - 1e-6):
            return True
        return (self.lost_top_drop <= 0
                or self._top_drop(site, radius=site.get('r', self.seen_radius))
                >= self.lost_top_drop - 1e-6)

    def _seen_through(self, tree) -> bool:
        v = self._seen_visits.get(tree['id'])
        return v is not None and len(v['done']) == 2 and max(v['done']) < self.seen_open

    def baseline_cb(self, msg: String):
        if not self.drone_baseline_ready:
            self.drone_baseline_ready = True
            self.get_logger().info(f'Drone canopy baseline received: {msg.data}')

    def coverage_cb(self, msg: String):
        match = re.search(r'coverage=([0-9.]+)%', msg.data)
        if match:
            self.coverage_pct = float(match.group(1)) / 100.0

    def survey_cb(self, msg: String):
        if self.loop_counter.update_from_status(msg.data) and not self.baseline_frozen:
            self.get_logger().info(
                f'survey loop {self.loop_counter.loops}/{self.freeze_min_loops} completed')

    # ── CHM fusion ───────────────────────────────────────────────────

    def _terrain_ground_offset(self) -> float:
        """Estimate the additive height offset of the terrain cloud.

        The depth camera's downward tilt means the cloud's ground is not at
        exactly 0 m (in practice 0.3-5 m). Where the canopy map shows known
        ground cells (< 1 m), the terrain heights at those cells measure the
        offset directly. Fall back to a low percentile of all terrain heights
        when there is no map/terrain overlap yet.
        """
        seen = self.terrain_hits > 0
        if not np.any(seen):
            return 0.0
        if self.canopy_data is not None:
            map_ground = (self.canopy_data >= 0) & (self.canopy_data < self.height_scale)
            rows, cols = np.nonzero(map_ground)
            if len(rows) >= 20:
                wx = self.canopy_ox + (cols + 0.5) * self.canopy_res
                wy = self.canopy_oy + (rows + 0.5) * self.canopy_res
                ix = ((wx - self.origin_x) / self.res).astype(np.int64)
                iy = ((wy - self.origin_y) / self.res).astype(np.int64)
                ok = (ix >= 0) & (ix < self.dim_x) & (iy >= 0) & (iy < self.dim_y)
                ix, iy = ix[ok], iy[ok]
                samples = self.terrain_height[ix, iy][seen[ix, iy]]
                if samples.size >= 20:
                    return float(np.percentile(samples, 25.0))
        heights = self.terrain_height[seen]
        return float(np.percentile(heights, 10.0))

    def _fused_chm(self):
        """Fuse the canopy map and terrain cloud into (chm, scanned).

        The canopy map is height above ground (ground about 0 m), so it is
        the primary source. The terrain cloud fills cells the map has
        not scanned, after subtracting its measured ground offset.
        """
        fused = np.zeros((self.dim_x, self.dim_y), dtype=np.float32)
        scanned = np.zeros((self.dim_x, self.dim_y), dtype=bool)

        # 1. Canopy map first (already height-above-ground).
        if self.canopy_data is not None and self.canopy_res is not None:
            # Sample the map at every CHM cell centre (backward mapping);
            # mapping map cells forward leaves stripes of holes when the two
            # resolutions differ.
            fx = self.origin_x + (np.arange(self.dim_x) + 0.5) * self.res
            fy = self.origin_y + (np.arange(self.dim_y) + 0.5) * self.res
            cols = np.floor((fx - self.canopy_ox) / self.canopy_res).astype(np.int64)
            rows = np.floor((fy - self.canopy_oy) / self.canopy_res).astype(np.int64)
            h, w = self.canopy_data.shape
            ok_x = (cols >= 0) & (cols < w)
            ok_y = (rows >= 0) & (rows < h)
            ix = np.nonzero(ok_x)[0]
            iy = np.nonzero(ok_y)[0]
            if len(ix) and len(iy):
                # canopy_data is [row=y, col=x]; the CHM is [ix, iy].
                sampled = self.canopy_data[np.ix_(rows[iy], cols[ix])].T
                seen = sampled >= 0
                block = np.where(seen, sampled / self.height_scale, 0.0)
                fused[np.ix_(ix, iy)] = block
                scanned[np.ix_(ix, iy)] = seen

        # 2. Terrain cloud fills the remaining holes, ground-normalised.
        # Cells seen fewer than terrain_min_hits times are mostly streaks from
        # single grazing rays, so they count as unknown.
        terrain_seen = self.terrain_hits >= self.terrain_min_hits
        if self.terrain_fill and np.any(terrain_seen):
            offset = self._terrain_ground_offset()
            fill = terrain_seen & ~scanned
            fused[fill] = np.clip(self.terrain_height[fill] - offset, 0.0, None)
            scanned[fill] = True
        return fused, scanned

    # ── Baseline / change handling ───────────────────────────────────

    def _candidate_det(self, cand):
        """The detection a freeze candidate enters the baseline with: its
        latest one, or with baseline_mean_position its position averaged
        over its last mean_position_hits hits."""
        det = cand[5]
        if not self.baseline_mean_position or len(cand[6]) < 2:
            return det
        xs, ys = zip(*cand[6])
        return dataclasses.replace(det, x=float(np.mean(xs)),
                                   y=float(np.mean(ys)))

    def _trim_candidate(self, cand, now_s):
        """Keep only the admission window: the last admit_window ticks, or
        with admit_window_s the ticks of the last admit_window_s seconds."""
        if self.admit_window_s > 0:
            start = next((k for k, t in enumerate(cand[7])
                          if now_s - t <= self.admit_window_s), len(cand[7]))
        else:
            start = max(0, len(cand[2]) - max(1, self.admit_window))
        cand[2] = cand[2][start:]
        cand[7] = cand[7][start:]

    def _keep_candidate(self, cand) -> bool:
        """Drop a candidate after candidate_max_misses misses in a row, or,
        with admit_window_s, once it has no hit left in the window."""
        if self.admit_window_s > 0:
            return any(cand[2])
        return cand[4] < self.candidate_max_misses

    def _candidate_confirmed(self, cand) -> bool:
        hits = sum(cand[2])
        if hits < self.baseline_stable_ticks:
            return False
        return (self.admit_fraction <= 0
                or hits >= self.admit_fraction * len(cand[2]))

    def _neighborhood_observed_fraction(self, det, scanned) -> float:
        """Gap-filled, crown-circle observed fraction (not raw hits).

        Thin stripes of missing LiDAR returns inside a flown-over crown are
        scan gaps, not unobserved ground, so the hit mask is morphologically
        closed first (same idea as the CHM pit fill). The check answers
        "did the drone fly over this crown?", not "did every cell return?".
        """
        r_px = max(1, int(round(det.radius_m / self.res)))
        ix = int((det.x - self.origin_x) / self.res)
        iy = int((det.y - self.origin_y) / self.res)
        x0, x1 = max(0, ix - r_px), min(self.dim_x, ix + r_px + 1)
        y0, y1 = max(0, iy - r_px), min(self.dim_y, iy + r_px + 1)
        window = scanned[x0:x1, y0:y1]
        if window.size == 0:
            return 0.0
        # Grid is [ix, iy]: rows carry x, columns carry y.
        rows, cols = np.mgrid[0:window.shape[0], 0:window.shape[1]]
        circle = (rows - (ix - x0)) ** 2 + (cols - (iy - y0)) ** 2 <= r_px ** 2
        # binary_closing treats the outside of the array as unobserved and
        # erodes the window border, where the crown circle touches it. Pad
        # with edge values, then crop back.
        iterations = max(1, int(self.itd_params.mask_close_iterations))
        padded = np.pad(window, iterations, mode='edge')
        closed = ndimage.binary_closing(
            padded,
            structure=np.ones((3, 3), dtype=bool),
            iterations=iterations)
        window = closed[iterations:-iterations, iterations:-iterations]
        return float(window[circle].mean()) if np.any(circle) else 0.0

    def _crown_coverage_detail(self, det, scanned):
        """Position, radius (m and cells), cells checked/observed, fraction."""
        r_px = max(1, int(round(det.radius_m / self.res)))
        ix = int((det.x - self.origin_x) / self.res)
        iy = int((det.y - self.origin_y) / self.res)
        x0, x1 = max(0, ix - r_px), min(self.dim_x, ix + r_px + 1)
        y0, y1 = max(0, iy - r_px), min(self.dim_y, iy + r_px + 1)
        window = scanned[x0:x1, y0:y1]
        rows, cols = np.mgrid[0:window.shape[0], 0:window.shape[1]]
        circle = (rows - (ix - x0)) ** 2 + (cols - (iy - y0)) ** 2 <= r_px ** 2
        gap_fraction = self._neighborhood_observed_fraction(det, scanned)
        return {
            'x': det.x, 'y': det.y,
            'radius_m': det.radius_m,
            'radius_cells': r_px,
            # Raw (not gap-filled) counts inside the crown circle; the
            # fraction is the gap-filled value the admission rule uses.
            'checked': int(circle.sum()),
            'observed': int(np.count_nonzero(window[circle])),
            'fraction': gap_fraction,
        }

    def _freeze_baseline(self, detections, scanned=None, chm=None):
        # A tree may enter the baseline only if it passes the same rules as a
        # gain: minimum crown area and >=80% of its crown neighbourhood
        # actually observed. Transient fragments cannot sneak into the
        # baseline and later fire a false "lost".
        eligible = [
            d for d in detections
            if d.area_m2 >= self.min_gain_area_m2
            and (scanned is None
                 or self._neighborhood_observed_fraction(d, scanned) >= 0.8)
        ]
        details = [dict(self._crown_coverage_detail(d, scanned),
                        area_m2=d.area_m2)
                   for d in detections if scanned is not None]
        for detail in details:
            reasons = []
            if detail['area_m2'] < self.min_gain_area_m2:
                reasons.append(f'area<{self.min_gain_area_m2}')
            if detail['fraction'] < 0.8:
                reasons.append('observed<0.8')
            self.get_logger().info(
                f'crown coverage ({detail["x"]:.1f},{detail["y"]:.1f}): '
                f'area={detail["area_m2"]:.2f}m2 '
                f'{"REJECT " + ",".join(reasons) if reasons else "ADMIT"} '
                f'r={detail["radius_m"]:.2f}m ({detail["radius_cells"]} cells), '
                f'checked={detail["checked"]}, observed={detail["observed"]}, '
                f'fraction={detail["fraction"]:.2f}')
            if detail['fraction'] < 0.8 and scanned is not None:
                r_px = detail['radius_cells']
                ix = int((detail['x'] - self.origin_x) / self.res)
                iy = int((detail['y'] - self.origin_y) / self.res)
                x0, x1 = max(0, ix - r_px), min(self.dim_x, ix + r_px + 1)
                y0, y1 = max(0, iy - r_px), min(self.dim_y, iy + r_px + 1)
                sub = scanned[x0:x1, y0:y1]
                step = max(1, sub.shape[0] // 24)
                rows = [''.join('#' if sub[i, j] else '.'
                                for j in range(0, sub.shape[1], step))
                        for i in range(0, sub.shape[0], step)]
                self.get_logger().info(
                    f'crown hit pattern (raw, {step}-cell bins):\n'
                    + '\n'.join(rows))
        if len(eligible) < max(1, self.min_baseline_trees):
            self.baseline_frozen = False
            self.get_logger().info(
                f'Baseline blocked: {len(eligible)} of {len(detections)} pass '
                f'admission, need {self.min_baseline_trees}',
                throttle_duration_sec=5.0)
            return False, (
                f'blocked: crown coverage, '
                f'{len(eligible)}/{len(detections)} candidates >=80% observed '
                f'(need {self.min_baseline_trees})')
        self.baseline_trees = [{
            'id': i,
            'x': d.x, 'y': d.y, 'height': d.height,
            'radius_m': d.radius_m, 'area_m2': d.area_m2,
        } for i, d in enumerate(eligible, start=1)]
        if self.freeze_median_ticks > 0:
            final = [(t['x'], t['y']) for t in self.baseline_trees]
            for tree in self.baseline_trees:
                tree['x'], tree['y'] = self._median_position(tree, final)
        canopy = pines = 0
        for tree in self.baseline_trees:
            seen = self._disc_mean(self.seen_map, tree)
            colour = self._disc_mean(self.species_mean, tree)
            tree['colour'] = colour
            is_canopy = seen is not None and seen >= self.seen_canopy
            is_pine = colour is not None and colour >= self.seen_pine
            tree['camera_pine'] = is_canopy and is_pine
            canopy += is_canopy
            pines += tree['camera_pine']
        self._seen_visits.clear()
        self._alerted_patches.clear()
        self._pine_patches = self._find_pine_patches() if self.camera_area_alerts else []
        attached = [t for t in self.baseline_trees if t.get('site')]
        admitted = self._admit_pine_patches()
        if self.seen_map is not None:
            self.get_logger().info(
                f'camera: {canopy} of {len(self.baseline_trees) - len(admitted)} baseline trees '
                f'seen as canopy, {pines} of them pine-coloured (camera LOST applies to these); '
                f'{len(self._pine_patches)} pine patches without a tree watched'
                + ''.join(f'; #{t["id"]} watched at its pine patch ({t["site"]["x"]:.1f}, '
                          f'{t["site"]["y"]:.1f}), {math.hypot(t["site"]["x"] - t["x"], t["site"]["y"] - t["y"]):.1f} m off'
                          for t in attached)
                + ''.join(f'; pine #{t["id"]} at ({t["x"]:.1f}, {t["y"]:.1f}) admitted: '
                          f'{t["site"]["cells"]}-cell pine patch, LiDAR tree in {t["hits"]} '
                          f'of the last {len(self._recent_dets)} ticks'
                          for t in admitted if not t.get('camera_only'))
                + (f'; {sum(1 for t in admitted if t.get("camera_only"))} camera-only '
                   f'pines admitted (patches >= {self.area_camera_cells} cells, no tree '
                   f'within {self.area_camera_clear:.1f} m)'
                   if any(t.get('camera_only') for t in admitted) else ''))
        self.next_tree_id = len(self.baseline_trees) + 1
        self.baseline_tree_count = len(self.baseline_trees)
        self.baseline_scanned = (scanned.copy()
                                 if scanned is not None else None)
        self.baseline_chm = (np.where(scanned, chm, -1.0).astype(np.float32)
                             if chm is not None and scanned is not None
                             else None)
        for tree in self.baseline_trees:     # own top on the latest map (merge_lost_crowns)
            top = (disc_max(self.baseline_chm, self.res, (self.origin_x, self.origin_y),
                            tree['x'], tree['y'], 0.75)
                   if self.baseline_chm is not None else None)
            tree['top'] = top if top is not None else tree['height']
        self.baseline_frozen = True
        self._alerted_lost_ids.clear()
        self._unmatched_streaks.clear()
        self._pending_lost.clear()
        self._reported_lost.clear()
        self.total_trees_lost = 0
        self.total_trees_gained = 0
        if self._plateau_time is not None:
            now = self.get_clock().now().nanoseconds / 1e9
            self.get_logger().info(
                f'plateau-to-freeze: {max(0.0, now - self._plateau_time):.1f}s')
        self.get_logger().info(
            f'Parrot tree baseline frozen: {len(self.baseline_trees)} '
            f'individual trees ({len(detections) - len(eligible)} rejected)')
        self.drone_baseline_pub.publish(String(
            data=f'frozen: {len(self.baseline_trees)} trees'))
        self._publish_baseline()
        return True, ''

    def reset_baseline_callback(self, request, response):
        fused, scanned = self._fused_chm()
        if not np.any(scanned):
            response.success = False
            response.message = 'No CHM data yet; cannot re-baseline.'
            return response
        detections, _, _ = detect_trees(
            fused, scanned, self.itd_params, self.origin_x, self.origin_y,
            return_debug=True)
        ok, reason = self._freeze_baseline(detections, scanned, fused)
        if not ok:
            response.success = False
            response.message = reason
            return response
        self.total_trees_lost = 0
        self.total_trees_gained = 0
        response.success = True
        response.message = (
            f'Baseline reset. {len(self.baseline_trees)} individual trees.')
        return response

    def _match_baseline(self, detections):
        """Optimal one-to-one matching of detections to baseline IDs.

        The association radius is adaptive per tree:
        ``min(0.4 * nearest_neighbour_distance, 1.5)`` m, so two trees 1.4 m
        apart can never swap IDs between passes. The assignment minimises
        total distance over all pairs (Hungarian algorithm), which is the
        correct one-to-one objective, unlike greedy nearest-neighbour.

        Returns ``(matched_ids, unmatched_base_indices)`` where matched_ids
        maps a detection index to its stable baseline ID (or None for new
        trees).
        """
        ids = [None] * len(detections)
        # Camera-only trees are never matched to a LiDAR detection.
        tracked = [i for i, t in enumerate(self.baseline_trees) if not t.get('camera_only')]
        camera_only = [i for i, t in enumerate(self.baseline_trees) if t.get('camera_only')]
        if not tracked or not detections:
            return ids, list(range(len(self.baseline_trees)))

        base = np.array([[self.baseline_trees[i]['x'], self.baseline_trees[i]['y']]
                         for i in tracked])
        det = np.array([[d.x, d.y] for d in detections])
        d_bb = np.hypot(
            base[:, None, 0] - base[None, :, 0],
            base[:, None, 1] - base[None, :, 1])
        np.fill_diagonal(d_bb, np.inf)
        nearest = d_bb.min(axis=1)
        radius = np.minimum(0.4 * nearest, 1.5)
        # A lone tree has no neighbour; use the configured fallback radius.
        radius = np.where(np.isinf(nearest), self.track_radius, radius)

        cost = np.hypot(
            det[:, None, 0] - base[None, :, 0],
            det[:, None, 1] - base[None, :, 1])
        allowed = cost <= radius[None, :]
        big = 1e6
        matrix = np.where(allowed, cost, big)
        rows, cols = linear_sum_assignment(matrix)

        unmatched = list(tracked)
        for ri, ci in zip(rows, cols):
            if matrix[ri, ci] < big:
                ids[ri] = self.baseline_trees[tracked[ci]]['id']
                unmatched.remove(tracked[ci])
        return ids, sorted(unmatched + camera_only)

    def _containing_crown(self, det):
        """Return the baseline tree whose crown contains det, else None."""
        best, best_d = None, float('inf')
        for tree in self.baseline_trees:
            if tree['id'] in self._alerted_lost_ids:
                continue  # a crown already reported lost can be regrown
            d = math.hypot(det.x - tree['x'], det.y - tree['y'])
            if d <= self.gain_crown_exclusion * tree['radius_m'] and d < best_d:
                best, best_d = tree, d
        return best

    def _lost_cells_near(self, tree, radius) -> int:
        """Count change-map lost cells within radius of the tree."""
        n = self._change_cells_near(self.change_lost, tree, radius)
        if self.lost_drop_evidence:
            n += self._change_cells_near(self.change_dropped, tree, radius)
        return n

    def _gained_cells_near(self, tree, radius) -> int:
        """Count change-map new-canopy cells within radius of the tree."""
        return self._change_cells_near(self.change_gained, tree, radius)

    def _top_drop(self, tree, mode: str = 'max', radius=None) -> float:
        """visibility.top_drop within `radius` (default lost_top_radius) of
        the tree: 'max' = baseline minus current disc max (m); 'frac' =
        fraction of the freeze-time top cells that fell >= 0.6 m. 0 if
        unknown."""
        if self.baseline_chm is None or self._current_chm is None:
            return 0.0
        return top_drop(self.baseline_chm, self._current_chm, self.res,
                        (self.origin_x, self.origin_y), tree['x'], tree['y'],
                        self.lost_top_radius if radius is None else radius, mode)

    def _evidence_radius(self, radius_m) -> float:
        return max(self.track_radius, min(radius_m, self.evidence_max_radius))

    def _change_cells_near(self, mask, tree, radius) -> int:
        if mask is None or not np.any(mask):
            return 0
        rows, cols = np.nonzero(mask)
        wx = self.change_ox + (cols + 0.5) * self.change_res
        wy = self.change_oy + (rows + 0.5) * self.change_res
        close = (wx - tree['x']) ** 2 + (wy - tree['y']) ** 2 <= radius ** 2
        return int(np.count_nonzero(close))

    # ── Publishing ───────────────────────────────────────────────────

    def publish_status(self):
        fused, scanned = self._fused_chm()
        if not np.any(scanned):
            return
        self._current_chm = np.where(scanned, fused, -1.0).astype(np.float32)
        det_chm = fused
        if self.baseline_mean_chm and not self.baseline_frozen:
            if self._chm_sum is None:
                self._chm_sum = np.zeros(fused.shape)
                self._chm_n = np.zeros(fused.shape)
            self._chm_sum[scanned] += fused[scanned]
            self._chm_n[scanned] += 1
            det_chm = np.where(self._chm_n > 0, self._chm_sum / np.maximum(self._chm_n, 1),
                               fused).astype(np.float32)
        detections, labels, canopy = detect_trees(
            det_chm, scanned, self.itd_params, self.origin_x, self.origin_y,
            return_debug=True)

        # Drop detections outside the surveyed box (fringe artifacts past the
        # patrol area).
        detections = [
            d for d in detections
            if self.survey_x_min <= d.x <= self.survey_x_max
            and self.survey_y_min <= d.y <= self.survey_y_max
        ]
        if self.colour_pines and not self.baseline_frozen:
            detections += self._colour_spires(fused, scanned, detections)
        if not self.baseline_frozen:
            self._recent_dets.append(list(detections))

        # Freeze the tree baseline once enough canopy has been observed.
        if not self.baseline_frozen:
            covered = int(np.count_nonzero(scanned))
            # coverage_pct and plateau_delta are fractions (0-1).
            if self.coverage_pct - self._last_cov < self.plateau_delta:
                self._cov_stable_ticks += 1
            else:
                self._cov_stable_ticks = 0
            self._last_cov = self.coverage_pct
            plateau = self._cov_stable_ticks >= self.plateau_ticks
            if plateau and not self._plateau_reached:
                self._plateau_reached = True
                self._plateau_time = self.get_clock().now().nanoseconds / 1e9
                if self.reset_candidates_at_plateau:
                    # Only count admission ticks after the plateau.
                    self._freeze_candidates = []
            if self._plateau_reached:
                self._ticks_since_plateau += 1

            # Adaptive association radius, identical to tracking:
            # min(0.4 * nearest-neighbour distance, 1.5 m).
            det_pos = np.array([[d.x, d.y] for d in detections])
            if len(det_pos) > 1:
                d_nn = np.hypot(
                    det_pos[:, None, 0] - det_pos[None, :, 0],
                    det_pos[:, None, 1] - det_pos[None, :, 1])
                np.fill_diagonal(d_nn, np.inf)
                radii = np.minimum(0.4 * d_nn.min(axis=1), 1.5)
                radii = np.where(np.isinf(radii), self.track_radius, radii)
            else:
                radii = np.full(len(det_pos), self.track_radius)

            # Per-tree stability: seen in baseline_stable_ticks of the last
            # admit_window ticks within the adaptive radius.
            now_s = self.get_clock().now().nanoseconds / 1e9
            seen = [False] * len(self._freeze_candidates)
            for det, radius in zip(detections, radii):
                best_idx, best_d = None, float('inf')
                for idx, cand in enumerate(self._freeze_candidates):
                    d = math.hypot(det.x - cand[0], det.y - cand[1])
                    if d <= cand[3] and d < best_d:
                        best_idx, best_d = idx, d
                if best_idx is None:
                    # [x, y, hit flags, radius, misses in a row, last det,
                    #  latest hit positions, tick times]
                    cand = [det.x, det.y, [1], float(radius), 0, det,
                            [(det.x, det.y)], [now_s]]
                    self._freeze_candidates.append(cand)
                    seen.append(True)
                else:
                    cand = self._freeze_candidates[best_idx]
                    (self._disp_post if self._plateau_reached
                     else self._disp_pre).append(best_d)
                    cand[0], cand[1], cand[3] = det.x, det.y, float(radius)
                    cand[5] = det
                    cand[2].append(1)
                    cand[6] = (cand[6] + [(det.x, det.y)])[
                        -self.mean_position_hits:]
                    cand[7].append(now_s)
                    cand[4] = 0
                    seen[best_idx] = True
            kept = []
            for i, cand in enumerate(self._freeze_candidates):
                if not seen[i]:
                    cand[2].append(0)
                    cand[7].append(now_s)
                    cand[4] += 1
                self._trim_candidate(cand, now_s)
                if self._keep_candidate(cand):
                    kept.append(cand)
            self._freeze_candidates = kept

            # _tick only advances after the freeze, so this log uses its own
            # counter.
            self._pre_freeze_ticks += 1
            if self._pre_freeze_ticks % 20 == 0 and \
                    (self._disp_pre or self._disp_post):
                for name, arr in (('pre-plateau', self._disp_pre),
                                  ('post-plateau', self._disp_post)):
                    if arr:
                        a = np.array(arr)
                        self.get_logger().info(
                            f'treetop displacement {name}: '
                            f'max={a.max():.2f}m p95={np.percentile(a, 95):.2f}m '
                            f'(n={len(a)})')

            # Freeze from every confirmed candidate, not only the trees
            # detected on this tick: a tree that drops out for the freeze
            # tick would otherwise be missing and come back as a false GAINED.
            stable_dets = [
                self._candidate_det(cand) for cand in self._freeze_candidates
                if self._candidate_confirmed(cand)]
            confirmed = len(stable_dets)
            # Candidates still being confirmed (the list is reset at the
            # plateau, so a tree that misses one early tick is not confirmed
            # yet) get up to freeze_max_wait_ticks before the freeze.
            pending = len(self._freeze_candidates) - confirmed
            waiting_for_pending = (
                pending > 0
                and self._ticks_since_plateau < self.freeze_max_wait_ticks)
            self.drone_baseline_pub.publish(String(data=(
                f'waiting: coverage {self.coverage_pct * 100:.1f}%, '
                f'plateau {self._cov_stable_ticks}/{self.plateau_ticks}, '
                f'candidates confirmed {confirmed}/{len(self._freeze_candidates)}'
                f'{f", {pending} pending" if pending else ""}, '
                f'min {self.min_baseline_trees}'
                f'{f", loop {self.loop_counter.loops}/{self.freeze_min_loops}" if self.freeze_min_loops else ""}')))
            if self.coverage_pct >= self.baseline_coverage and \
                    self.loop_counter.loops >= self.freeze_min_loops and \
                    covered >= self.min_baseline_cells and \
                    len(stable_dets) >= self.min_baseline_trees and \
                    self._cov_stable_ticks >= self.plateau_ticks and \
                    not waiting_for_pending:
                if pending:
                    self.get_logger().info(
                        f'freezing without {pending} unconfirmed candidate(s) '
                        f'after {self._ticks_since_plateau} ticks')
                ok, reason = self._freeze_baseline(stable_dets, scanned, fused)
                if not ok:
                    self.drone_baseline_pub.publish(String(data=reason))
                    self._publish_positions(detections)
                    return
            else:
                # Detections are published during the survey too (dashboard,
                # and replays can see why a tree missed the baseline).
                self._publish_positions(detections)
                return

        if not self.baseline_frozen:
            self._publish_positions(detections)
            return

        matched_ids, unmatched = self._match_baseline(detections)
        if self.seen_lost:
            self._update_seen(self.get_clock().now().nanoseconds / 1e9)
            if self.camera_area_alerts:
                self._check_pine_patches()

        # Pre-hysteresis dropout rate: how many baseline trees are unmatched
        # on a single tick, before the lost/gained streak filters decide
        # whether anything is actually reported.
        self._tick += 1
        if self._tick % 5 == 0:
            self.get_logger().info(
                f'pre-hysteresis: {len(unmatched)}/{self.baseline_tree_count} '
                f'trees unmatched this tick')
        if self._tick % 50 == 0:
            self.get_logger().info(
                f'gains skipped (unobserved at baseline): '
                f'{self._gains_skipped_unobserved}, '
                f'rejected (inside a crown): {self._gains_rejected_in_crown}, '
                f'rejected (no new-canopy evidence): '
                f'{self._gains_rejected_no_evidence}')

        # Reset the "still missing" streak for every tree seen this tick.
        seen_ids = {tid for tid in matched_ids if tid is not None}
        for tid in seen_ids:
            self._unmatched_streaks.pop(tid, None)

        # New trees (GAINED): only promote a detection to a stable ID after
        # several consecutive ticks in the same cell, so one-off flicker
        # cannot inflate the tree count.
        for di, tid in enumerate(matched_ids):
            if tid is not None:
                continue
            det = detections[di]
            if det.area_m2 < self.min_gain_area_m2:
                continue
            # Confirm the same treetop for several consecutive ticks, allowing
            # normal centroid jitter (up to the association radius).
            slot = None
            for idx, (px, py, count) in enumerate(self._pending_gains):
                if math.hypot(det.x - px, det.y - py) <= self.track_radius:
                    slot = idx
                    break
            if slot is None:
                self._pending_gains.append([det.x, det.y, 1])
                continue
            px, py, count = self._pending_gains[slot]
            self._pending_gains[slot] = [
                (px * count + det.x) / (count + 1),
                (py * count + det.y) / (count + 1),
                count + 1,
            ]
            if self._pending_gains[slot][2] < self.gain_stable_ticks:
                continue
            self._pending_gains.pop(slot)
            # A detection inside an existing tree's crown is its treetop
            # jumping to another lobe (big oaks have several), not a new tree.
            host = self._containing_crown(det)
            if host is not None:
                self._gains_rejected_in_crown += 1
                self.get_logger().info(
                    f'gain rejected: ({det.x:.1f},{det.y:.1f}) is inside the '
                    f'crown of tree #{host["id"]} at '
                    f'({host["x"]:.1f},{host["y"]:.1f}), '
                    f'r={host["radius_m"]:.1f}m',
                    throttle_duration_sec=10.0)
                continue
            # A real gain needs its whole crown-radius neighbourhood to have
            # been observed at baseline. Anything else is newly-discovered
            # area, logged as such and never emitted as a change event.
            cell_ix = int((det.x - self.origin_x) / self.res)
            cell_iy = int((det.y - self.origin_y) / self.res)
            if self.baseline_scanned is not None:
                r_px = max(1, int(round(det.radius_m / self.res)))
                x0, x1 = max(0, cell_ix - r_px), min(self.dim_x, cell_ix + r_px + 1)
                y0, y1 = max(0, cell_iy - r_px), min(self.dim_y, cell_iy + r_px + 1)
                window = self.baseline_scanned[x0:x1, y0:y1]
                if window.size == 0 or window.mean() < 0.8:
                    self._gains_skipped_unobserved += 1
                    self.get_logger().info(
                        f'newly observed (not a change event): '
                        f'({det.x:.1f},{det.y:.1f})',
                        throttle_duration_sec=10.0)
                    continue
            evidence = self._gained_cells_near(
                {'x': det.x, 'y': det.y}, self._evidence_radius(det.radius_m))
            if evidence < self.gain_evidence_cells:
                self._gains_rejected_no_evidence += 1
                self.get_logger().info(
                    f'gain rejected: ({det.x:.1f},{det.y:.1f}) has '
                    f'{evidence} new-canopy cells nearby '
                    f'(need {self.gain_evidence_cells})',
                    throttle_duration_sec=10.0)
                continue
            new_id = self.next_tree_id
            self.next_tree_id += 1
            self.baseline_trees.append({
                'id': new_id,
                'x': det.x, 'y': det.y, 'height': det.height,
                'radius_m': det.radius_m, 'area_m2': det.area_m2,
            })
            matched_ids[di] = new_id
            self.total_trees_gained += 1
            self._publish_event('GAINED', new_id, det.x, det.y, det.area_m2)
        self._prev_dets = [(d.x, d.y) for d in detections]

        # Lost trees: require a persistent miss AND a minimum amount of
        # canopy-loss evidence near the tree before declaring it lost. A
        # single noisy change-map cell or one tick of centroid jitter cannot
        # trigger a loss.
        for bi in unmatched:
            tree = self.baseline_trees[bi]
            if tree['id'] in self._alerted_lost_ids:
                continue
            radius = self._evidence_radius(tree['radius_m'])
            height_evidence = (not tree.get('camera_only')
                               and self._lost_cells_near(tree, radius) >= self.lost_evidence_cells
                               and self._top_fell(tree))
            # The drop is measured over the camera's (smaller) disc: a
            # neighbour's branch inside the 1 m disc can hold its max.
            camera_evidence = (
                self.seen_lost and self._seen_through(tree)
                and self._site_dropped(tree.get('site') or tree))
            if not (height_evidence or camera_evidence):
                self._unmatched_streaks.pop(tree['id'], None)
                continue
            streak = self._unmatched_streaks.get(tree['id'], 0) + 1
            self._unmatched_streaks[tree['id']] = streak
            if streak < self.lost_streak_threshold:
                continue
            self._alerted_lost_ids.add(tree['id'])
            tree['lost_by'] = 'height' if height_evidence else 'camera'
            if not height_evidence:
                where = 'its pine patch' if tree.get('site') else 'the spot'
                self.get_logger().info(
                    f'LOST #{tree["id"]} on the camera: it now sees through '
                    f'{where}, the height map shows no loss')
            now = self.get_clock().now().nanoseconds / 1e9
            if self.merge_lost_crowns:
                tree['hold'] = (max(self.merge_window_s, self.merge_hold_s)
                                if self._has_standing_host(tree) else self.merge_window_s)
                self._pending_lost.append((now, tree))
            else:
                self._report_lost(tree, now)
        if self.merge_lost_crowns:
            self._flush_pending_lost(self.get_clock().now().nanoseconds / 1e9)

        status = TreeStatus()
        status.header = Header(
            stamp=self.get_clock().now().to_msg(), frame_id='parrot1_odom')
        status.baseline_count = self.baseline_tree_count   # frozen count, like tree_mapper
        status.current_count = len(seen_ids)
        status.lost_count = self.total_trees_lost
        status.gained_count = self.total_trees_gained
        self.status_pub.publish(status)

        self._publish_positions(detections)
        self._publish_treetop_markers(detections)
        self._publish_crown_map(labels, canopy, fused)

    def _report_lost(self, tree, when):
        self.total_trees_lost += 1
        self._reported_lost.append((when, tree))
        self._publish_event('LOST', tree['id'], tree['x'], tree['y'], tree['area_m2'])
        self._publish_change_marker(tree, lost=True)
        self._publish_note({'type': 'LOST', 'id': tree['id'], 'x': tree['x'], 'y': tree['y'],
                            'by': tree.get('lost_by', 'height'), 'source': self._source(tree)})

    def _is_oak(self, tree) -> bool:
        colour = tree.get('colour')
        if colour is not None:
            return colour < self.seen_pine
        return tree.get('top', tree.get('height', 0.0)) >= self.merge_min_height

    def _has_standing_host(self, tree) -> bool:
        """A non-pine crown within merge_max_dist, at least 1 / merge_hold_ratio
        times this one's area, that has not gone LOST."""
        if self.merge_hold_s <= self.merge_window_s or not self._is_oak(tree):
            return False
        return any(
            other is not tree and other['id'] not in self._alerted_lost_ids
            and self._is_oak(other)
            and tree['area_m2'] <= self.merge_hold_ratio * other['area_m2']
            and math.hypot(other['x'] - tree['x'], other['y'] - tree['y']) <= self.merge_max_dist
            for other in self.baseline_trees)

    def _merge_host(self, tree, when):
        """A larger oak crown that went LOST near this one at about the same
        time, or None."""
        if not self._is_oak(tree):
            return None
        reported = {id(o) for _, o in self._reported_lost}
        for t, other in self._reported_lost + self._pending_lost:
            if (other is not tree
                    and -tree.get('hold', self.merge_window_s) <= when - t
                    <= max(self.merge_window_s, self.merge_after_s)
                    and self._is_oak(other)
                    and math.hypot(other['x'] - tree['x'], other['y'] - tree['y'])
                    <= self.merge_max_dist):
                larger = (other['area_m2'], -other['id']) > (tree['area_m2'], -tree['id'])
                after = (self.merge_after_ratio > 0 and id(other) in reported and when > t
                         and other['area_m2'] >= self.merge_after_ratio * tree['area_m2'])
                if larger or after:
                    return other
        return None

    def _flush_pending_lost(self, now):
        """Publish (or merge) the held LOSTs whose window has passed, largest first."""
        due = [e for e in self._pending_lost
               if now - e[0] >= e[1].get('hold', self.merge_window_s)]
        if not due:
            return
        for when, tree in sorted(due, key=lambda e: -e[1]['area_m2']):
            host = self._merge_host(tree, when)
            if host is None:
                self._report_lost(tree, when)
            else:
                d = math.hypot(host['x'] - tree['x'], host['y'] - tree['y'])
                self.get_logger().info(
                    f'LOST #{tree["id"]} merged into #{host["id"]}: part of the same '
                    f'felled crown ({d:.1f} m)')
                self._publish_note({'type': 'MERGED', 'id': tree['id'], 'into': host['id'],
                                    'x': tree['x'], 'y': tree['y'], 'dist': round(d, 1)})
        self._pending_lost = [e for e in self._pending_lost
                              if now - e[0] < e[1].get('hold', self.merge_window_s)]

    # ── Message builders ─────────────────────────────────────────────

    def _publish_event(self, event_type, tree_id, x, y, area_m2):
        event = TreeChangeEvent()
        event.header = Header(
            stamp=self.get_clock().now().to_msg(), frame_id='parrot1_odom')
        event.event_type = event_type
        event.tree_id = int(tree_id)
        event.x = float(x)
        event.y = float(y)
        event.area_m2 = float(area_m2)
        self.event_pub.publish(event)
        self.get_logger().info(
            f'Parrot tree event: {event_type} #{tree_id} at ({x:.1f}, {y:.1f})')

    def _publish_change_marker(self, tree, lost):
        marker = Marker()
        marker.header = Header(
            stamp=self.get_clock().now().to_msg(), frame_id='parrot1_odom')
        marker.id = tree['id']
        marker.ns = 'parrot_tree_lost' if lost else 'parrot_tree_gained'
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose = Pose(
            position=Point(x=tree['x'], y=tree['y'], z=2.5),
            orientation=Quaternion(w=1.0))
        size = max(tree['area_m2'] ** 0.5, 1.0)
        marker.scale.x = size
        marker.scale.y = size
        marker.scale.z = 5.0
        marker.color.r = 1.0 if lost else 0.0
        marker.color.g = 0.0
        marker.color.b = 0.0 if lost else 1.0
        marker.color.a = 0.9
        self.marker_pub.publish(MarkerArray(markers=[marker]))

    def _publish_baseline(self):
        pts = np.array([[t['x'], t['y'], float(t['id'])]
                        for t in self.baseline_trees], dtype=np.float32)
        self.baseline_pub.publish(self._xyz_cloud(pts.reshape(-1, 3)))
        info = {
            'trees': [{'id': t['id'], 'x': round(t['x'], 2), 'y': round(t['y'], 2),
                       'area_m2': round(t['area_m2'], 1), 'source': self._source(t),
                       'species': self._species(t)} for t in self.baseline_trees],
            'pine_patches': [{'x': round(p['x'], 2), 'y': round(p['y'], 2), 'cells': p['cells']}
                             for p in self._pine_patches],
        }
        self.baseline_info_pub.publish(String(data=json.dumps(info)))

    @staticmethod
    def _source(tree) -> str:
        """'lidar', 'camera' (camera-only pine) or 'lidar+camera' (a pine
        patch the LiDAR also saw, admitted at the freeze)."""
        if tree.get('camera_only'):
            return 'camera'
        return 'lidar+camera' if 'hits' in tree else 'lidar'

    def _species(self, tree):
        """'pine' / 'oak' from the camera colour at the freeze, else None."""
        if tree.get('camera_pine') or tree.get('site'):
            return 'pine'
        colour = tree.get('colour')
        if colour is None:
            return None
        return 'pine' if colour >= self.seen_pine else 'oak'

    def _publish_note(self, note):
        self.change_note_pub.publish(String(data=json.dumps(note)))

    def _xyz_cloud(self, pts):
        cloud = PointCloud2()
        cloud.header = Header(
            stamp=self.get_clock().now().to_msg(), frame_id='parrot1_odom')
        cloud.height = 1
        cloud.width = len(pts)
        cloud.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        cloud.is_bigendian = False
        cloud.point_step = 12
        cloud.row_step = cloud.point_step * cloud.width
        cloud.is_dense = True
        cloud.data = pts.astype(np.float32).tobytes()
        return cloud

    def _publish_positions(self, detections):
        pts = np.array([[d.x, d.y, 0.0] for d in detections], dtype=np.float32)
        cloud = PointCloud2()
        cloud.header = Header(
            stamp=self.get_clock().now().to_msg(), frame_id='parrot1_odom')
        cloud.height = 1
        cloud.width = len(pts)
        cloud.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        cloud.is_bigendian = False
        cloud.point_step = 12
        cloud.row_step = cloud.point_step * cloud.width
        cloud.is_dense = True
        cloud.data = pts.tobytes()
        self.positions_pub.publish(cloud)

    def _publish_treetop_markers(self, detections):
        markers = MarkerArray()
        for i, det in enumerate(detections):
            marker = Marker()
            marker.header = Header(
                stamp=self.get_clock().now().to_msg(), frame_id='parrot1_odom')
            marker.id = i
            marker.ns = 'treetop'
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose = Pose(
                position=Point(x=det.x, y=det.y, z=det.height),
                orientation=Quaternion(w=1.0))
            marker.scale.x = 0.6
            marker.scale.y = 0.6
            marker.scale.z = 0.6
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.9
            markers.markers.append(marker)
        self.treetop_pub.publish(markers)

    def _publish_crown_map(self, labels, canopy, fused):
        data = np.where(canopy, labels, 0).astype(np.int8)
        grid = OccupancyGrid()
        grid.header = Header(
            stamp=self.get_clock().now().to_msg(), frame_id='parrot1_odom')
        grid.info.resolution = self.res
        grid.info.width = self.dim_x
        grid.info.height = self.dim_y
        grid.info.origin = Pose(
            position=Point(x=self.origin_x, y=self.origin_y, z=0.0),
            orientation=Quaternion(w=1.0))
        grid.data = data.T.flatten().tolist()
        self.crown_pub.publish(grid)

    def destroy_node(self):
        self.get_logger().info(
            f'Parrot Tree Tracker shutting down. {self.terrain_points} terrain '
            f'points. Lost: {self.total_trees_lost}, Gained: {self.total_trees_gained}.'
        )
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ParrotTreeTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
