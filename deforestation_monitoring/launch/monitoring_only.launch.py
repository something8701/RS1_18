#!/usr/bin/env python3
# monitoring_only.launch.py: Monitoring-only launch
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
