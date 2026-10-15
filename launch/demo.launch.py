from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from naro.paths import load_case


def launch_demo(context):
    name = LaunchConfiguration('case').perform(context)
    root, case = load_case(name)
    body = case['body_odom']
    import numpy as np
    lever = (np.asarray(body)[:3, :3].T @
             (np.asarray(case['sensor_origin']) - np.asarray(case['here']))).tolist()
    share = Path(get_package_share_directory('naro'))
    return [
        Node(package='naro', executable='node', name='naro', output='screen',
             parameters=[str(share / 'config' / 'reloc.yaml'),
                         {'map_dir': str(root / case['map']), 'lidar_translation': lever}]),
        Node(package='naro', executable='play_scan', name='scan_player', output='screen',
             parameters=[{'case': name}]),
        Node(package='rviz2', executable='rviz2', arguments=['-d', str(share / 'rviz' / 'demo.rviz')],
             condition=IfCondition(LaunchConfiguration('rviz'))),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('case', default_value='courtyard'),
        DeclareLaunchArgument('rviz', default_value='true'),
        OpaqueFunction(function=launch_demo),
    ])
