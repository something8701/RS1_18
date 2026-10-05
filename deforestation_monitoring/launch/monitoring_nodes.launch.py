#!/usr/bin/env python3
"""
The monitoring node set, in one place. demo_full.launch.py (with the
simulation) and monitoring_only.launch.py (against a running simulation)
both include it.

Arguments:
    drone        (true)  drone nodes (scan_mapper, pattern_scanner, demo_patrol,
                         drone_mapper, parrot_tree_tracker, camera nodes)
    husky        (true)  Husky nodes (tree_mapper, mission_coordinator,
                         husky_patrol, tree_fusion, simulate_deforestation,
                         evaluate_mission, Husky camera republisher)
    world        (dense_forest)  must match the simulated world; sets the
                         survey boxes and the dense-only settings
    remove_trees (true for dense_forest)  cut trees once the baseline freezes
    camera_lost  (true)  camera LOST rules in the tracker
    camera_pines (false) add pine tips found by camera colour as trees
    use_sim_time (true)  use simulation /clock time
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    drone_enabled = LaunchConfiguration('drone', default='true')
    husky_enabled = LaunchConfiguration('husky', default='true')
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world_name = LaunchConfiguration('world', default='dense_forest')

    declare_drone = DeclareLaunchArgument(
        'drone', default_value='true',
        description='Launch drone-only nodes'
    )
    declare_husky = DeclareLaunchArgument(
        'husky', default_value='true',
        description='Launch Husky-only nodes'
    )
    declare_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use simulation /clock time'
    )
    declare_world = DeclareLaunchArgument(
        'world', default_value='dense_forest',
        description='Gazebo world name. Must match the world launched by the simulation.'
    )
    # Tree removal is a change test. The no-change gates on sparse_trees /
    # cluster_test must never remove anything, so it defaults to on only for
    # dense_forest. Override with remove_trees:=true/false.
    remove_trees = LaunchConfiguration('remove_trees')
    declare_remove_trees = DeclareLaunchArgument(
        'remove_trees',
        default_value=PythonExpression(
            ["'true' if '", world_name, "' == 'dense_forest' else 'false'"]),
        description='Delete target trees after the tree baseline freezes'
    )
    # Drone camera cues for the tracker (option A). camera_lost turns on the
    # camera LOST rules, which catch pines cut from under oak crowns.
    # camera_pines stays off: it adds false trees at oak crown edges.
    camera_lost = LaunchConfiguration('camera_lost')
    declare_camera_lost = DeclareLaunchArgument(
        'camera_lost', default_value='true',
        description='Tracker: the camera seeing through a tree\'s spot counts as LOST evidence'
    )
    camera_pines = LaunchConfiguration('camera_pines')
    declare_camera_pines = DeclareLaunchArgument(
        'camera_pines', default_value='false',
        description='Tracker: add pine tips confirmed by camera colour as trees'
    )

    # Coverage box: sparse_trees and cluster_test only have trees within
    # about ±15 m and ±10 m; dense_forest uses ±30 m.
    coverage_x_min = PythonExpression([
        "'-15.0' if '", world_name, "' == 'sparse_trees' else "
        "('-10.0' if '", world_name, "' == 'cluster_test' else '-30.0')"
    ])
    coverage_x_max = PythonExpression([
        "'15.0' if '", world_name, "' == 'sparse_trees' else "
        "('10.0' if '", world_name, "' == 'cluster_test' else '30.0')"
    ])
    coverage_y_min = PythonExpression([
        "'-15.0' if '", world_name, "' == 'sparse_trees' else "
        "('-10.0' if '", world_name, "' == 'cluster_test' else '-30.0')"
    ])
    coverage_y_max = PythonExpression([
        "'15.0' if '", world_name, "' == 'sparse_trees' else "
        "('10.0' if '", world_name, "' == 'cluster_test' else '30.0')"
    ])

    # In the dense forest the tracker keeps trees out to ±37 m (80 of the 214
    # trees stand beyond ±30 m). Camera pine patches stay within ±30 m
    # (area_edge_margin): near the scan edge the camera sees only obliquely.
    tracker_half = PythonExpression([
        "'37.0' if '", world_name, "' == 'dense_forest' else "
        "('15.0' if '", world_name, "' == 'sparse_trees' else "
        "('10.0' if '", world_name, "' == 'cluster_test' else '30.0'))"
    ])
    tracker_half_neg = PythonExpression(["'-' + '", tracker_half, "'"])
    area_edge_margin = PythonExpression([
        "'7.0' if '", world_name, "' == 'dense_forest' else '0.0'"
    ])

    # Dense forest: freeze both baselines (scan_mapper's change baseline and
    # the tracker's trees) only after two full survey loops, so every cell
    # has a second pass. Other worlds freeze after one.
    survey_loops = PythonExpression([
        "'2' if '", world_name, "' == 'dense_forest' else '0'"
    ])

    # The drone patrol box shrinks for the small scenario worlds so test
    # cases finish surveying quickly.
    patrol_bound = PythonExpression([
        "'16.6' if '", world_name, "' == 'cluster_test' else "
        "('21.6' if '", world_name, "' == 'sparse_trees' else '36.6')"
    ])
    patrol_bound_neg = PythonExpression([
        "'-16.6' if '", world_name, "' == 'cluster_test' else "
        "('-21.6' if '", world_name, "' == 'sparse_trees' else '-36.6')"
    ])
    # Dense forest: the lanes run along x; turn at ±39 m (past the edge trees,
    # the ground ends at ±37.5 m) so the trees near the lane ends are flown
    # over at full speed. The lanes' y positions are unchanged.
    patrol_x_bound = PythonExpression([
        "'39.0' if '", world_name, "' == 'dense_forest' else '", patrol_bound, "'"
    ])
    patrol_x_bound_neg = PythonExpression(["'-' + '", patrol_x_bound, "'"])

    # --- Node: Tree Mapper (Husky lidar → tree trunk map) ---
    tree_mapper = Node(
        package='deforestation_monitoring',
        executable='tree_mapper',
        name='tree_mapper',
        output='screen',
        condition=IfCondition(husky_enabled),
        remappings=[
            ('/tf', '/husky1/tf'),
            ('/tf_static', '/husky1/tf_static'),
        ],
        parameters=[{
            'resolution': 0.25,
            'map_size_x': 80.0,
            'map_size_y': 80.0,
            'publish_rate': 1.0,
            'use_sim_time': use_sim_time,
        }],
    )

    # --- Node: Canopy Scan Mapper (drone pitched LiDAR → 2.5D canopy map) ---
    scan_mapper = Node(
        package='deforestation_monitoring',
        executable='scan_mapper',
        name='scan_mapper',
        output='screen',
        condition=IfCondition(drone_enabled),
        parameters=[{
            'resolution': 0.25,
            'map_size_x': 80.0,
            'map_size_y': 80.0,
            'scan_topic': '/parrot1/scan',
            'odom_topic': '/parrot1/odometry',
            'map_frame': 'parrot1_odom',
            'sensor_pitch': 1.5708,   # 90° down, as in parrot.gazebo.xacro
            'sensor_yaw': 0.0,
            'altitude': 10.0,         # sim odometry reports z=0, so altitude is fixed
            'canopy_threshold': 2.0,
            'height_scale': 10.0,     # map value = 0-10 m canopy height → 0-100
            'baseline_threshold': 20,
            'baseline_loop_scans': 15,
            'coverage_required': 0.9,
            # height-drop evidence for the tracker: a tree removed beside a
            # taller neighbour falls onto its crown, not to ground
            'drop_evidence_m': 1.0,
            # cells under-seen at the snapshot keep building their change
            # baseline
            'baseline_fill': True,
            'baseline_min_loops': survey_loops,
            'coverage_x_min': coverage_x_min,
            'coverage_x_max': coverage_x_max,
            'coverage_y_min': coverage_y_min,
            'coverage_y_max': coverage_y_max,
            'publish_rate': 1.0,
            'use_sim_time': use_sim_time,
        }],
    )

    # --- Node: Pattern Scanner (canopy change → /suspicious_areas flags) ---
    pattern_scanner = Node(
        package='deforestation_monitoring',
        executable='pattern_scanner',
        name='pattern_scanner',
        output='screen',
        condition=IfCondition(drone_enabled),
        parameters=[{
            # Flag canopy loss from /canopy_change_map.
            'detection_mode': 'change',
            'lost_value_threshold': -50.0,
            'min_lost_cells': 8,
            'max_change_flags': 5,
            # Height-mode heuristics, kept off: on the height map any
            # positive gap threshold floods GAP flags.
            'gap_threshold': 0,
            'min_gap_area': 8,
            'anomaly_height_drop': 60.0,
            'line_min_cells': 1000,
            'scan_rate': 0.5,
            'detection_frame': 'parrot1_odom',  # canopy map now comes from the drone's scan_mapper
            'use_sim_time': use_sim_time,
        }],
    )

    # --- Node: Mission Coordinator (flags → priority queue → Nav2 dispatch) ---
    mission_coordinator = Node(
        package='deforestation_monitoring',
        executable='mission_coordinator',
        name='mission_coordinator',
        output='screen',
        condition=IfCondition(husky_enabled),
        remappings=[
            ('/tf', '/husky1/tf'),
            ('/tf_static', '/husky1/tf_static'),
        ],
        parameters=[{
            'priority_mode': 'composite',
            'dedup_window': 300.0,
            'max_retries': 3,
            'requeue_penalty': 50.0,
            'use_sim_time': use_sim_time,
        }],
    )

    # --- Node: Drone Patrol (lawnmower survey + smart flag response) ---
    demo_patrol = Node(
        package='deforestation_monitoring',
        executable='demo_patrol',
        name='demo_patrol',
        output='screen',
        condition=IfCondition(drone_enabled),
        parameters=[{
            'patrol_x_min': patrol_x_bound_neg,
            'patrol_x_max': patrol_x_bound,
            'patrol_y_min': patrol_bound_neg,
            'patrol_y_max': patrol_bound,
            'strip_spacing': 9.0,
            'altitude': 10.0,
            'dedup_window': 300.0,
            'use_sim_time': use_sim_time,
        }],
    )

    # --- Node: Drone Terrain Mapper (camera → terrain PointCloud2) ---
    drone_mapper = Node(
        package='deforestation_monitoring',
        executable='drone_mapper',
        name='drone_mapper',
        output='screen',
        remappings=[
            ('/tf', '/parrot1/tf'),
            ('/tf_static', '/parrot1/tf_static'),
        ],
        condition=IfCondition(drone_enabled),
        parameters=[{
            'publish_rate': 2.0,
            'depth_topic': '/parrot1/camera/depth/points',
            'target_frame': 'parrot1_odom',
            'altitude': 10.0,
            'terrain_resolution': 0.15,
            'max_publish_points': 50000,
            'downsample_step': 4,
            'process_every_n_frames': 1,
            'use_sim_time': use_sim_time,
        }],
    )

    # --- Node: camera species mapper (drone camera colour on the map grid) ---
    camera_species = Node(
        package='deforestation_monitoring',
        executable='camera_species_mapper',
        name='camera_species_mapper',
        output='screen',
        condition=IfCondition(drone_enabled),
        parameters=[{'altitude': 10.0, 'use_sim_time': use_sim_time}],
    )

    # --- Node: Parrot Tree Tracker (fused CHM -> individual tree IDs) ---
    parrot_tree_tracker = Node(
        package='deforestation_monitoring',
        executable='parrot_tree_tracker',
        name='parrot_tree_tracker',
        output='screen',
        condition=IfCondition(drone_enabled),
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('deforestation_monitoring'),
                'config',
                'tree_detection_params.yaml',
            ]),
            {
                'canopy_map_topic': '/forest_canopy_map',
                'change_map_topic': '/canopy_change_map',
                'baseline_status_topic': '/drone_baseline_status',
                'drone_terrain_topic': '/drone_terrain',
                'height_scale': 10.0,
                'min_baseline_cells': 150,
                'min_baseline_trees': 2,
                'baseline_coverage': 0.9,   # tree baseline only after >=90% coverage
                'track_radius': 0.6,
                'lost_streak_threshold': 3,
                # LOST: for 3 ticks, >= 6 loss cells nearby and the tree's top
                # gone (max within 1 m fell >= 0.3 m with >= 90 % of its top
                # cells down >= 0.6 m, or the mean fell >= 2 m)
                'lost_evidence_cells': 6,
                'lost_drop_evidence': True,
                'lost_top_drop': 0.3,
                'lost_top_frac': 0.9,
                'lost_mean_drop': 2.0,
                'terrain_fill': False,
                'seen_lost': PythonExpression(["'", camera_lost, "' == 'true'"]),
                # Area alerts and patch sites need the camera, so they follow
                # camera_lost.
                'baseline_mean_chm': True,
                'merge_lost_crowns': True,
                'freeze_min_loops': survey_loops,
                'freeze_median_ticks': 30,
                'freeze_median_radius': 2.0,
                'freeze_median_exclusive': True,
                'merge_hold_s': 240.0,
                'merge_after_ratio': 0.6,
                'camera_area_alerts': PythonExpression(["'", camera_lost, "' == 'true'"]),
                'area_attach': PythonExpression(["'", camera_lost, "' == 'true'"]),
                'colour_pines': PythonExpression(["'", camera_pines, "' == 'true'"]),
                'survey_x_min': tracker_half_neg,
                'survey_x_max': tracker_half,
                'survey_y_min': tracker_half_neg,
                'survey_y_max': tracker_half,
                'area_edge_margin': area_edge_margin,
                'publish_rate': 1.0,
                'use_sim_time': use_sim_time,
            },
        ],
    )

    # --- Node: Tree Fusion (lidar + camera → combined confidence map) ---
    tree_fusion = Node(
        package='deforestation_monitoring',
        executable='tree_fusion',
        name='tree_fusion',
        output='screen',
        condition=IfCondition(husky_enabled),
        parameters=[{
            'resolution': 0.25,
            'map_size_x': 80.0,
            'map_size_y': 80.0,
            'fusion_frame': 'husky1_map',
            'use_sim_time': use_sim_time,
        }],
    )

    # --- Node: Husky Survey Patrol (Nav2 lawnmower grid) ---
    husky_patrol = Node(
        package='deforestation_monitoring',
        executable='husky_patrol',
        name='husky_patrol',
        output='screen',
        condition=IfCondition(husky_enabled),
        remappings=[
            ('/tf', '/husky1/tf'),
            ('/tf_static', '/husky1/tf_static'),
        ],
        # The grid starts inside the initial SLAM map (about 28 x 30 m around
        # the spawn): Nav2 rejects goals off the global costmap. SLAM grows
        # the map as the Husky patrols.
        parameters=[{
            'grid_x_min': -10.0,
            'grid_x_max': 10.0,
            'grid_y_min': -10.0,
            'grid_y_max': 10.0,
            'strip_spacing': 6.0,
            'use_sim_time': use_sim_time,
        }],
    )

    # --- Node: Tree Removal (deletes a tree cluster in Gazebo + ground truth) ---
    tree_removal = Node(
        package='deforestation_monitoring',
        executable='simulate_tree_removal',
        name='simulate_tree_removal',
        output='screen',
        condition=IfCondition(PythonExpression(
            ["'", drone_enabled, "' == 'true' and '", remove_trees, "' == 'true'"])),
        parameters=[{
            'world': world_name,       # must match the world the sim launched
            'use_center_radius': False,
            # dense_forest trees inside ±30 m, each >= 4.8 m from its nearest
            # neighbour, so each loss is one clean crown
            'tree_names': ['oak_78', 'pine_93', 'oak_64', 'pine_75'],
            'frame_id': 'parrot1_odom',
            'require_both_baselines': False,
            'use_sim_time': use_sim_time,
        }],
    )

    # --- Node: Mission Evaluation (scores detections against ground truth) ---
    evaluate_mission = Node(
        package='deforestation_monitoring',
        executable='evaluate_mission',
        name='evaluate_mission',
        output='screen',
        condition=IfCondition(drone_enabled),
        parameters=[{
            'match_radius': 8.0,
            'summary_period': 10.0,
            'use_sim_time': use_sim_time,
        }],
    )

    # --- Rosbridge WebSocket (for live dashboard) ---
    rosbridge = Node(
        package='rosbridge_server',
        executable='rosbridge_websocket',
        name='rosbridge_websocket',
        output='screen',
        parameters=[{'port': 9090}],
    )

    # --- Data Server (stores history, serves dashboard) ---
    data_server = Node(
        package='deforestation_monitoring',
        executable='data_server',
        name='data_server',
        output='screen',
    )

    # --- Clock watchdog (detects Gazebo /clock stalls) ---
    clock_watchdog = Node(
        package='deforestation_monitoring',
        executable='clock_watchdog',
        name='clock_watchdog',
        output='screen',
        parameters=[{'stall_timeout': 10.0}],
    )

    # --- Compressed image republishers (for dashboard camera feeds) ---
    # Our own camera_republisher: image_transport's republish ignores the
    # `out` remap in Humble, so its topic names cannot be set.
    drone_cam_compressed = Node(
        package='deforestation_monitoring',
        executable='camera_republisher',
        name='drone_cam_republish',
        output='screen',
        parameters=[{
            'in_topic': '/parrot1/camera/image',
            'out_topic': '/parrot1/camera/compressed',
            'jpeg_quality': 70,
            'max_rate': 5.0,
            'use_sim_time': use_sim_time,
        }],
        condition=IfCondition(drone_enabled),
    )
    husky_cam_compressed = Node(
        package='deforestation_monitoring',
        executable='camera_republisher',
        name='husky_cam_republish',
        output='screen',
        parameters=[{
            'in_topic': '/husky1/camera/image',
            'out_topic': '/husky1/camera/compressed',
            'jpeg_quality': 70,
            'max_rate': 5.0,
            'use_sim_time': use_sim_time,
        }],
        condition=IfCondition(husky_enabled),
    )

    # --- Staggered start: let the simulation initialise first ---
    delayed_drone_mapper = TimerAction(period=15.0, actions=[drone_mapper])
    delayed_camera_species = TimerAction(period=15.0, actions=[camera_species])
    delayed_parrot_tree_tracker = TimerAction(period=20.0, actions=[parrot_tree_tracker])
    delayed_scan_mapper = TimerAction(period=15.0, actions=[scan_mapper])
    delayed_tree_mapper = TimerAction(period=15.0, actions=[tree_mapper])
    delayed_scanner = TimerAction(period=15.0, actions=[pattern_scanner])
    delayed_coordinator = TimerAction(period=15.0, actions=[mission_coordinator])
    delayed_patrol = TimerAction(period=25.0, actions=[demo_patrol])
    delayed_husky = TimerAction(period=35.0, actions=[husky_patrol])
    delayed_fusion = TimerAction(period=20.0, actions=[tree_fusion])
    delayed_tree_removal = TimerAction(period=30.0, actions=[tree_removal])
    delayed_eval = TimerAction(period=15.0, actions=[evaluate_mission])

    return LaunchDescription([
        declare_drone,
        declare_husky,
        declare_sim_time,
        declare_world,
        declare_remove_trees,
        declare_camera_lost,
        declare_camera_pines,
        rosbridge,
        data_server,
        clock_watchdog,
        drone_cam_compressed,
        husky_cam_compressed,
        delayed_drone_mapper,
        delayed_camera_species,
        delayed_parrot_tree_tracker,
        delayed_scan_mapper,
        delayed_tree_mapper,
        delayed_scanner,
        delayed_coordinator,
        delayed_patrol,
        delayed_husky,
        delayed_fusion,
        delayed_tree_removal,
        delayed_eval,
    ])
