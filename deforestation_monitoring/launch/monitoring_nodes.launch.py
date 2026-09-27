#!/usr/bin/env python3
"""
All deforestation monitoring nodes (perception, communication, decision,
UI). Included by demo_full.launch.py (with the simulation) and by
monitoring_only.launch.py (with a simulation that is already running), so
both use the same node set.

Arguments:
    drone       (true)  Drone nodes (scan_mapper, pattern_scanner,
                        demo_patrol, drone_mapper, drone camera republisher)
    husky       (true)  Husky nodes (tree_mapper, mission_coordinator,
                        husky_patrol, tree_fusion, simulate_deforestation,
                        evaluate_mission, husky camera republisher)
    use_sim_time (true) Use simulation /clock time
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
        description='Gazebo world name; must match the world the simulation launched.'
    )
    # Tree removal is a change test. The no-change tests on sparse_trees and
    # cluster_test must never remove anything, so the default is on only for
    # dense_forest. Override with remove_trees:=true/false.
    remove_trees = LaunchConfiguration('remove_trees')
    declare_remove_trees = DeclareLaunchArgument(
        'remove_trees',
        default_value=PythonExpression(
            ["'true' if '", world_name, "' == 'dense_forest' else 'false'"]),
        description='Delete target trees after the tree baseline freezes'
    )

    # sparse_trees and cluster_test only have ground and trees within about
    # 15 m and 10 m of the centre, so coverage is measured on that smaller
    # box. dense_forest uses 30 m.
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

    # The drone patrol box is smaller for the small test worlds so the
    # survey finishes quickly.
    patrol_bound = PythonExpression([
        "'16.6' if '", world_name, "' == 'cluster_test' else "
        "('21.6' if '", world_name, "' == 'sparse_trees' else '36.6')"
    ])
    patrol_bound_neg = PythonExpression([
        "'-16.6' if '", world_name, "' == 'cluster_test' else "
        "('-21.6' if '", world_name, "' == 'sparse_trees' else '-36.6')"
    ])

    # Tree mapper (Husky lidar to tree trunk map)
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

    # Canopy scan mapper (drone LiDAR to canopy height map)
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
            'sensor_pitch': 1.5708,   # 90 degrees down, as in parrot.gazebo.xacro
            'sensor_yaw': 0.0,
            'altitude': 10.0,         # fixed, because sim odometry reports z=0
            'canopy_threshold': 2.0,
            'height_scale': 10.0,     # 0-10 m canopy height maps to 0-100
            'baseline_threshold': 20,
            'coverage_required': 0.9,
            'coverage_x_min': coverage_x_min,
            'coverage_x_max': coverage_x_max,
            'coverage_y_min': coverage_y_min,
            'coverage_y_max': coverage_y_max,
            'publish_rate': 1.0,
            'use_sim_time': use_sim_time,
        }],
    )

    # Pattern scanner (canopy change to /suspicious_areas flags)
    pattern_scanner = Node(
        package='deforestation_monitoring',
        executable='pattern_scanner',
        name='pattern_scanner',
        output='screen',
        condition=IfCondition(drone_enabled),
        parameters=[{
            # Default mode: canopy-loss regions from /canopy_change_map.
            'detection_mode': 'change',
            'lost_value_threshold': -50.0,
            'min_lost_cells': 8,
            'max_change_flags': 5,
            # Older height-mode heuristics, switched off: on the height map
            # ground reads about 1 and the canopy/ground boundary differs by
            # up to about 79, so any positive threshold floods GAP flags.
            'gap_threshold': 0,
            'min_gap_area': 8,
            'anomaly_height_drop': 60.0,
            'line_min_cells': 1000,
            'scan_rate': 0.5,
            'detection_frame': 'parrot1_odom',  # frame of the drone scan_mapper map
            'use_sim_time': use_sim_time,
        }],
    )

    # Mission coordinator (flags to priority queue to Nav2)
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

    # Drone patrol (lawnmower survey, reacts to flags)
    demo_patrol = Node(
        package='deforestation_monitoring',
        executable='demo_patrol',
        name='demo_patrol',
        output='screen',
        condition=IfCondition(drone_enabled),
        parameters=[{
            'patrol_x_min': patrol_bound_neg,
            'patrol_x_max': patrol_bound,
            'patrol_y_min': patrol_bound_neg,
            'patrol_y_max': patrol_bound,
            'strip_spacing': 9.0,
            'altitude': 10.0,
            'dedup_window': 300.0,
            'use_sim_time': use_sim_time,
        }],
    )

    # Drone terrain mapper (depth camera to terrain PointCloud2)
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

    # Parrot tree tracker (CHM to individual tree IDs)
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
                'baseline_coverage': 0.9,   # tree baseline only after 90% coverage
                'track_radius': 0.6,
                'lost_streak_threshold': 3,
                'lost_evidence_cells': 10,   # dense tests: removed trees 13-150, standing <= 6
                'survey_x_min': coverage_x_min,
                'survey_x_max': coverage_x_max,
                'survey_y_min': coverage_y_min,
                'survey_y_max': coverage_y_max,
                'publish_rate': 1.0,
                'use_sim_time': use_sim_time,
            },
        ],
    )

    # Tree fusion (lidar and camera to one confidence map)
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

    # Husky survey patrol (Nav2 lawnmower grid)
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
        # The grid starts inside the first SLAM map (about 28.5 x 30.7 m
        # around the spawn). Nav2 rejects waypoints outside the global
        # costmap and the robot would never move, so the map would never
        # grow. As the Husky patrols, SLAM extends the map.
        parameters=[{
            'grid_x_min': -10.0,
            'grid_x_max': 10.0,
            'grid_y_min': -10.0,
            'grid_y_max': 10.0,
            'strip_spacing': 6.0,
            'use_sim_time': use_sim_time,
        }],
    )

    # Tree removal (deletes trees in Gazebo and publishes ground truth)
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
            # dense_forest trees inside the 30 m tracker survey box, each at
            # least 4.8 m from its nearest neighbour, so each loss is one
            # clean crown.
            'tree_names': ['oak_78', 'pine_93', 'oak_64', 'pine_75'],
            'frame_id': 'parrot1_odom',
            'require_both_baselines': False,
            'use_sim_time': use_sim_time,
        }],
    )

    # Mission evaluation (scores detections against ground truth)
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

    # Rosbridge websocket (for the live dashboard)
    rosbridge = Node(
        package='rosbridge_server',
        executable='rosbridge_websocket',
        name='rosbridge_websocket',
        output='screen',
        parameters=[{'port': 9090}],
    )

    # Data server (stores history, serves the dashboard)
    data_server = Node(
        package='deforestation_monitoring',
        executable='data_server',
        name='data_server',
        output='screen',
    )

    # Clock watchdog (detects Gazebo /clock stalls)
    clock_watchdog = Node(
        package='deforestation_monitoring',
        executable='clock_watchdog',
        name='clock_watchdog',
        output='screen',
        parameters=[{'stall_timeout': 10.0}],
    )

    # Compressed image republishers for the dashboard camera feeds.
    # image_transport's republish ignores the `out` remap in Humble (the
    # topic stays /out/compressed), so our own camera_republisher is used.
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

    # Start nodes with a delay so the simulation can initialise first
    delayed_drone_mapper = TimerAction(period=15.0, actions=[drone_mapper])
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
        rosbridge,
        data_server,
        clock_watchdog,
        drone_cam_compressed,
        husky_cam_compressed,
        delayed_drone_mapper,
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
