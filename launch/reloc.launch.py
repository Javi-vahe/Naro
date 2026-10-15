from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = str(Path(get_package_share_directory('naro')) / 'config' / 'reloc.yaml')
    defaults = dict(map_dir='', config=config)
    return LaunchDescription([
        *[DeclareLaunchArgument(key, default_value=value) for key, value in defaults.items()],
        Node(package='naro', executable='node', name='naro', output='screen',
             parameters=[LaunchConfiguration('config'), {'map_dir': LaunchConfiguration('map_dir')}]),
    ])
