#!/usr/bin/env python3
# showcase_demo.launch.py: Client showcase: survey the showcase forest, remove 4 trees, detect them
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('n_trees', default_value='4'),
        DeclareLaunchArgument('lite', default_value='false',
                              description='Lighter run for slower PCs'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([PathJoinSubstitution([
                FindPackageShare('deforestation_monitoring'), 'launch',
                'removal_test.launch.py'])]),
            launch_arguments={
                'world': 'showcase_forest',
                'n_trees': LaunchConfiguration('n_trees'),
                'balanced': 'true',
                'lite': LaunchConfiguration('lite'),
            }.items(),
        ),
    ])
