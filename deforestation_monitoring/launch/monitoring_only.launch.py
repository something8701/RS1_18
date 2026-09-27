#!/usr/bin/env python3
"""
Monitoring only. Use this when the simulation is already running.

    ros2 launch deforestation_monitoring monitoring_only.launch.py

Launches the same monitoring nodes as demo_full.launch.py (both include
monitoring_nodes.launch.py) without the simulation, for a Gazebo instance
where the Husky and drone are already spawned. Robots can be turned off
with drone:=false or husky:=false.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    declare_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use simulation /clock time'
    )

    monitoring_nodes = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('deforestation_monitoring'),
                'launch',
                'monitoring_nodes.launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time,
        }.items(),
    )

    return LaunchDescription([
        declare_sim_time,
        LogInfo(msg='=== Deforestation Monitoring (nodes only) ==='),
        LogInfo(msg='Simulation assumed already running. Dashboard at http://localhost:8081'),
        monitoring_nodes,
    ])
