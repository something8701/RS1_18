#!/usr/bin/env python3
"""Client showcase: survey the showcase forest, remove 4 trees, detect them.

    ros2 launch deforestation_monitoring showcase_demo.launch.py
    ros2 launch deforestation_monitoring showcase_demo.launch.py n_trees:=10
    ros2 launch deforestation_monitoring showcase_demo.launch.py lite:=true

Runs removal_test.launch.py on world showcase_forest (every tree visible
from above) with oak/pine targets alternating. The drone surveys to >= 90%
coverage and freezes the tree baseline, 4 trees are deleted from Gazebo,
the drone keeps patrolling for 2 survey loops, and the report comparing the
detections with the simulator ground truth is written to
/tmp/deforestation_eval/removal_test_<stamp>.md (verdict on
/removal_test_result). Dashboard: http://localhost:8081
"""

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
