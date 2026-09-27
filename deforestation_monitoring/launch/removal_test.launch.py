#!/usr/bin/env python3
"""Live tree removal acceptance test.

Starts the full demo (drone only, no scripted tree removal) and the
removal_test node. The node waits for 90% coverage and the frozen tree
baseline, deletes 10 trees, keeps patrolling for 2 survey loops and writes
a report to /tmp/deforestation_eval/removal_test_<stamp>.md.

    ros2 launch deforestation_monitoring removal_test.launch.py
    ros2 launch deforestation_monitoring removal_test.launch.py n_trees:=5
    ros2 topic echo /removal_test_result
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _box(world, sparse, cluster, other):
    return PythonExpression([
        f"'{sparse}' if '", world, "' == 'sparse_trees' else "
        f"('{cluster}' if '", world, f"' == 'cluster_test' else '{other}')"])


def generate_launch_description():
    world = LaunchConfiguration('world')
    n_trees = LaunchConfiguration('n_trees')
    settle_loops = LaunchConfiguration('settle_loops')
    balanced = LaunchConfiguration('balanced')

    demo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([PathJoinSubstitution([
            FindPackageShare('deforestation_monitoring'), 'launch',
            'demo_full.launch.py'])]),
        launch_arguments={'world': world, 'remove_trees': 'false'}.items(),
    )

    test = Node(
        package='deforestation_monitoring',
        executable='removal_test',
        name='removal_test',
        output='screen',
        parameters=[{
            'world': world,
            'n_trees': n_trees,
            'settle_loops': settle_loops,
            'balanced': PythonExpression(["'", balanced, "' == 'true'"]),
            'min_coverage': 0.9,
            'survey_x_min': _box(world, -15.0, -10.0, -30.0),
            'survey_x_max': _box(world, 15.0, 10.0, 30.0),
            'survey_y_min': _box(world, -15.0, -10.0, -30.0),
            'survey_y_max': _box(world, 15.0, 10.0, 30.0),
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='dense_forest'),
        DeclareLaunchArgument('n_trees', default_value='10'),
        DeclareLaunchArgument('settle_loops', default_value='2'),
        DeclareLaunchArgument('balanced', default_value='false',
                              description='Remove oaks and pines alternately'),
        demo,
        TimerAction(period=20.0, actions=[test]),
    ])
