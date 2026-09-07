#!/usr/bin/env python3
"""
Monitoring node set — the single source of truth for the deforestation
pipeline (perception → communication → decision → UI). Included by
demo_full.launch.py (with the simulation) and monitoring_only.launch.py
(against an already-running simulation) so the two can never drift apart.

Arguments:
    drone       (true)  Launch drone-only nodes (scan_mapper, pattern_scanner,
                        demo_patrol, drone_mapper, drone camera republisher)
    husky       (true)  Launch Husky-only nodes (tree_mapper, mission_coordinator,
                        husky_patrol, tree_fusion, simulate_deforestation,
                        evaluate_mission, husky camera republisher)
    use_sim_time (true) Use simulation /clock time
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    drone_enabled = LaunchConfiguration('drone', default='true')
    husky_enabled = LaunchConfiguration('husky', default='true')
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

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
            'sensor_pitch': 1.5708,   # 90° down — matches WSL parrot.gazebo.xacro edit
            'sensor_yaw': 0.0,
            'altitude': 10.0,         # sim odometry reports z=0 — altitude is a fixed param
            'canopy_threshold': 2.0,
            'height_scale': 10.0,     # map value = 0-10 m canopy height → 0-100
            'baseline_threshold': 20,
            'baseline_loop_scans': 15,
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
            # Default mode: canopy-loss detection on /canopy_change_map —
            # the real deforestation signature, no false-flag flooding.
            'detection_mode': 'change',
            'lost_value_threshold': -50.0,
            'min_lost_cells': 8,
            'max_change_flags': 5,
            # Legacy height-mode heuristics (still gated — height-map
            # ground reads ≈1 and canopy/ground boundaries differ by up
            # to ~79, so any positive threshold floods GAP flags).
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
            'patrol_x_min': -15.0,
            'patrol_x_max': 15.0,
            'patrol_y_min': -15.0,
            'patrol_y_max': 15.0,
            'strip_spacing': 8.0,
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
        condition=IfCondition(drone_enabled),
        parameters=[{
            'publish_rate': 2.0,
            'altitude': 10.0,
            'fov_width': 15.0,
            'use_sim_time': use_sim_time,
        }],
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
        # The grid starts INSIDE the initial SLAM map (which only covers
        # ~28.5x30.7 m around the spawn). Waypoints outside the global
        # costmap are rejected by Nav2 ("goal off the global costmap") and
        # the robot never moves, so the map can never grow. As the Husky
        # patrols, SLAM grows the map toward the flagged sites.
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
        condition=IfCondition(drone_enabled),
        parameters=[{
            'world': 'simple_trees',   # must match the world the sim launched
            'use_center_radius': False,
            'tree_names': ['pine_9', 'oak_10', 'pine_11', 'pine_13'],
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
    # Uses our own camera_republisher node: image_transport's republish
    # ignores the `out` remap in Humble (topic stays /out/compressed), so
    # the launch cannot rely on it for exact topic names.
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
        rosbridge,
        data_server,
        clock_watchdog,
        drone_cam_compressed,
        husky_cam_compressed,
        delayed_drone_mapper,
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
