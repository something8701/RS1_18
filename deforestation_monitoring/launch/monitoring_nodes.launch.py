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
    world  (dense_forest) Gazebo world the simulation launched. Selects which
                        trees simulate_tree_removal deletes.
    husky_x/husky_y/husky_yaw (0, 0, 0), parrot_x/parrot_y/parrot_yaw (2, 0, 0)
                        Spawn poses of the two robots in the Gazebo world
                        (metres / radians). Used to publish the static
                        transform husky1_map -> parrot1_odom so drone flags
                        can be transformed into the Husky's Nav2 frame.
                        Defaults match 41068_ignition.launch.py.
"""

import math

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# Trees simulate_tree_removal deletes, per world. Each set is a real
# cluster inside the drone's +/-15 m patrol area, so the clearing is
# actually surveyed before and after removal.
REMOVAL_TREES = {
    'simple_trees': ['pine_9', 'oak_10', 'pine_11', 'pine_13'],
    'dense_forest': ['pine_90', 'pine_91', 'pine_105', 'oak_106'],  # ~(-3, -7)
}


def _world_dependent_nodes(context, *args, **kwargs):
    """Nodes whose configuration depends on resolved launch arguments."""
    lc = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731
    world = lc('world')
    drone = lc('drone').lower() == 'true'
    husky = lc('husky').lower() == 'true'
    use_sim_time = lc('use_sim_time').lower() == 'true'
    actions = []

    # --- Tree removal (deletes a tree cluster in Gazebo + ground truth) ---
    if drone:
        trees = REMOVAL_TREES.get(world)
        if trees is None:
            print(f'[monitoring_nodes] No removal trees defined for world '
                  f'"{world}" — simulate_tree_removal not started.')
        else:
            tree_removal = Node(
                package='deforestation_monitoring',
                executable='simulate_tree_removal',
                name='simulate_tree_removal',
                output='screen',
                parameters=[{
                    'world': world,
                    'use_center_radius': False,
                    'tree_names': trees,
                    'frame_id': 'parrot1_odom',
                    # With the Husky running, wait for BOTH baselines so the
                    # trees are never removed before the drone has mapped them.
                    'require_both_baselines': husky,
                    'use_sim_time': use_sim_time,
                }],
            )
            actions.append(TimerAction(period=30.0, actions=[tree_removal]))

    # --- Static TF linking the drone's frame into the Husky's TF tree ---
    # husky1_map is anchored at the Husky spawn pose (SLAM Toolbox) and
    # parrot1_odom at the drone spawn pose (Gazebo OdometryPublisher), so
    # T(husky1_map -> parrot1_odom) = inverse(husky spawn) * parrot spawn.
    if drone and husky:
        hx, hy, hyaw = (float(lc(n)) for n in ('husky_x', 'husky_y', 'husky_yaw'))
        px, py, pyaw = (float(lc(n)) for n in ('parrot_x', 'parrot_y', 'parrot_yaw'))
        dx, dy = px - hx, py - hy
        x = math.cos(hyaw) * dx + math.sin(hyaw) * dy
        y = -math.sin(hyaw) * dx + math.cos(hyaw) * dy
        yaw = pyaw - hyaw
        actions.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='husky_map_to_parrot_odom',
            output='screen',
            arguments=['--x', f'{x:.4f}', '--y', f'{y:.4f}', '--z', '0',
                       '--yaw', f'{yaw:.6f}', '--pitch', '0', '--roll', '0',
                       '--frame-id', 'husky1_map',
                       '--child-frame-id', 'parrot1_odom'],
            remappings=[('/tf', '/husky1/tf'),
                        ('/tf_static', '/husky1/tf_static')],
            parameters=[{'use_sim_time': use_sim_time}],
        ))
    return actions


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
    declare_world = DeclareLaunchArgument(
        'world', default_value='dense_forest',
        description='Gazebo world (selects the trees simulate_tree_removal deletes)'
    )
    # Defaults match the spawn poses in 41068_ignition_bringup's
    # 41068_ignition.launch.py: Husky (0, 0, yaw 0), Parrot (2, 0, yaw 0).
    # Update both places together if either spawn pose changes.
    spawn_defaults = {'husky_x': '0.0', 'husky_y': '0.0', 'husky_yaw': '0.0',
                      'parrot_x': '2.0', 'parrot_y': '0.0', 'parrot_yaw': '0.0'}
    declare_spawn = [
        DeclareLaunchArgument(n, default_value=v,
                              description=f'Robot spawn pose component {n} (world frame)')
        for n, v in spawn_defaults.items()
    ]

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
            'publish_rate': 1.0,
            # Swath cloud for RViz/dashboard: 20k points instead of 200k.
            # rosbridge JSON-encodes this cloud every second for the
            # dashboard, which at 200k points costs a lot of CPU.
            'max_cloud_points': 20000,
            # Baseline + change detection now live in change_detector.
            'enable_change_detection': False,
            'use_sim_time': use_sim_time,
        }],
    )

    # --- Node: Change Detector (canopy baseline vs current → height loss) ---
    change_detector = Node(
        package='deforestation_monitoring',
        executable='change_detector',
        name='change_detector',
        output='screen',
        condition=IfCondition(drone_enabled),
        parameters=[{
            'height_scale': 10.0,          # must match scan_mapper
            'canopy_threshold': 2.0,
            # 1.5 m, not 3.0: the drone measures crowns in simple_trees at only
            # ~3-6 m, so a 3 m drop made most removed trees undetectable.
            'height_drop_threshold': 1.5,
            'persistence_updates': 3,      # map publishes at 1 Hz → 3 s
            # Drone lidar is 2 Hz at up to 3 m/s → scan lines ~1.5 m apart.
            # Must exceed that spacing (lower it if the lidar rate is raised).
            'gap_bridge_m': 2.0,
            'min_cluster_cells': 8,
            'baseline_trigger': 'survey',  # freeze after first lawnmower pass
            'survey_coverage_pct': 100.0,
            'region_x_min': -15.0, 'region_x_max': 15.0,   # = demo_patrol grid
            'region_y_min': -15.0, 'region_y_max': 15.0,
            'baseline_file': '/tmp/deforestation_eval/canopy_baseline.npz',
            'save_baseline': True,
            'load_baseline': False,
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
        # Logs every 0.5-1 s at INFO; only show warnings and errors.
        arguments=['--ros-args', '--log-level', 'warn'],
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
        # Logs every 0.5-1 s at INFO; only show warnings and errors.
        arguments=['--ros-args', '--log-level', 'warn'],
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
    delayed_change_detector = TimerAction(period=15.0, actions=[change_detector])
    delayed_eval = TimerAction(period=15.0, actions=[evaluate_mission])

    return LaunchDescription([
        declare_drone,
        declare_husky,
        declare_sim_time,
        declare_world,
        *declare_spawn,
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
        delayed_change_detector,
        OpaqueFunction(function=_world_dependent_nodes),
        delayed_eval,
    ])
