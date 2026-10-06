#!/usr/bin/env python3
"""
Full Deforestation Monitoring Demo
====================================
Launches the simulation with both robots, SLAM, Nav2, RViz, then the whole
monitoring node set (tree mapper, pattern scanner, mission coordinator,
drone patrol, simulated deforestation events, evaluation, dashboard).

The monitoring nodes live in monitoring_nodes.launch.py, which
monitoring_only.launch.py includes too.

Usage:
    ros2 launch deforestation_monitoring demo_full.launch.py

For a slower PC:
    ros2 launch deforestation_monitoring demo_full.launch.py lite:=true

lite:=true turns off the Gazebo window, RViz and the drone camera in one go
(the dashboard at http://localhost:8081 still works). Each can also be set on
its own: gui:=false, rviz:=false, camera:=false. Without the camera the
tracker uses the height map only, so pines cut from under oak crowns are
missed more often.
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # --- Arguments ---
    drone_enabled = LaunchConfiguration('drone', default='true')
    use_husky = LaunchConfiguration('use_husky', default='false')
    husky_enabled = use_husky
    world_name = LaunchConfiguration('world', default='dense_forest')
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    remove_trees = LaunchConfiguration('remove_trees')
    lite = LaunchConfiguration('lite')
    gui = LaunchConfiguration('gui')
    rviz = LaunchConfiguration('rviz')
    camera = LaunchConfiguration('camera')
    # gui / rviz / camera default to on, or to off with lite:=true
    not_lite = PythonExpression(["'false' if '", lite, "'.lower() in ['true', '1'] else 'true'"])
    declare_lite = DeclareLaunchArgument(
        'lite', default_value='false',
        description='Lighter run for slower PCs: gui, rviz and camera off'
    )
    declare_gui = DeclareLaunchArgument(
        'gui', default_value=not_lite,
        description='Open the Gazebo window (on unless lite:=true)'
    )
    declare_rviz = DeclareLaunchArgument(
        'rviz', default_value=not_lite,
        description='Open RViz (on unless lite:=true)'
    )
    declare_camera = DeclareLaunchArgument(
        'camera', default_value=not_lite,
        description='Simulate and process the drone camera (on unless lite:=true)'
    )

    declare_drone = DeclareLaunchArgument(
        'drone', default_value='true',
        description='Launch the Parrot drone'
    )
    declare_husky = DeclareLaunchArgument(
        'use_husky', default_value='false',
        description='Launch the Husky ground rover (off by default to reduce '
                    'sim load; the code stays so it can be re-enabled)'
    )
    declare_world = DeclareLaunchArgument(
        'world', default_value='dense_forest',
        description='Gazebo world to use (dense_forest = 214 trees, 75 x 75 m)'
    )
    declare_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use simulation /clock time'
    )
    declare_remove_trees = DeclareLaunchArgument(
        'remove_trees',
        default_value=PythonExpression(
            ["'true' if '", world_name, "' == 'dense_forest' else 'false'"]),
        description='Delete target trees after the tree baseline freezes '
                    '(default: only in dense_forest)'
    )

    # --- Include the main 41068 simulation ---
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('41068_ignition_bringup'),
                'launch',
                '41068_ignition.launch.py'
            ])
        ]),
        launch_arguments={
            'husky': husky_enabled,
            'parrot': drone_enabled,
            'slam': 'true',
            'nav2': 'true',
            'rviz': rviz,
            'gui': gui,
            'parrot_camera': camera,
            'world': world_name,
        }.items(),
    )

    # --- Include the monitoring node set (perception → decision → UI) ---
    monitoring_nodes = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('deforestation_monitoring'),
                'launch',
                'monitoring_nodes.launch.py'
            ])
        ]),
        launch_arguments={
            'drone': drone_enabled,
            'husky': husky_enabled,
            'use_sim_time': use_sim_time,
            'world': world_name,
            'remove_trees': remove_trees,
            'camera': camera,
        }.items(),
    )

    return LaunchDescription([
        declare_drone,
        declare_husky,
        declare_world,
        declare_sim_time,
        declare_remove_trees,
        declare_lite,
        declare_gui,
        declare_rviz,
        declare_camera,
        LogInfo(msg='=== Deforestation Monitoring Demo ==='),
        LogInfo(msg=['Gazebo window: ', gui, ', RViz: ', rviz, ', drone camera: ', camera]),
        LogInfo(msg='Starting simulation (Husky: Nav2, Drone: cmd_vel patrol)...'),
        sim_launch,
        monitoring_nodes,
        LogInfo(msg='Simulation launched. Dashboard at http://localhost:8081'),
        LogInfo(msg='Full demo pipeline activated.'),
        LogInfo(msg='Watch RViz for:'),
        LogInfo(msg='  - /forest_canopy_map (drone LiDAR canopy heights, 0-10 m)'),
        LogInfo(msg='  - /canopy_change_map (red = canopy lost, blue = new canopy)'),
        LogInfo(msg='  - /drone_lidar_points (pitched-LiDAR push-broom swaths)'),
        LogInfo(msg='  - /forest_trunk_map (Husky lidar trunk detections)'),
        LogInfo(msg='  - /forest_change_map (Husky trunk diff: red = trees lost)'),
        LogInfo(msg='  - /fused_tree_map (green=both sensors, yellow=lidar, blue=camera)'),
        LogInfo(msg='  - /canopy_change_markers + /tree_change_markers'),
        LogInfo(msg='  - /fusion_markers (green=both, yellow=lidar-only, blue=camera-only)'),
        LogInfo(msg='  - /suspicious_markers (orange/red/purple cylinders)'),
        LogInfo(msg='  - Husky navigating to flagged sites'),
        LogInfo(msg='  - /inspection_reports in terminal output'),
        LogInfo(msg=''),
        LogInfo(msg='Change detection:'),
        LogInfo(msg='  ros2 topic echo /canopy_change_events  (drone canopy)'),
        LogInfo(msg='  ros2 topic echo /forest_change_events  (Husky trunks)'),
        LogInfo(msg='  ros2 service call /scan_mapper/reset_baseline std_srvs/srv/Trigger'),
        LogInfo(msg='  ros2 service call /tree_mapper/reset_baseline std_srvs/srv/Trigger'),
        LogInfo(msg=''),
        LogInfo(msg='Priority modes (change via param):'),
        LogInfo(msg='  ros2 param set /mission_coordinator priority_mode composite'),
        LogInfo(msg='  Modes: closest, confidence, area, severity, composite'),
        LogInfo(msg=''),
        LogInfo(msg='Manual commands:'),
        LogInfo(msg='  ros2 topic echo /suspicious_areas'),
        LogInfo(msg='  ros2 topic echo /inspection_reports'),
        LogInfo(msg='  ros2 topic echo /mission_status'),
        LogInfo(msg='  ros2 topic echo /survey_status'),
        LogInfo(msg='  ros2 topic echo /drone_baseline_status'),
        LogInfo(msg='  ros2 topic echo /baseline_status'),
        LogInfo(msg='Evaluation (accuracy / false positives / mission time):'),
        LogInfo(msg='  ros2 topic echo /evaluation_summary'),
        LogInfo(msg='  CSV: /tmp/deforestation_eval/evaluation.csv'),
    ])
