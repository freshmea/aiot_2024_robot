import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    param_dir = LaunchConfiguration('param_dir',
                                    default=os.path.join(
                                        get_package_share_directory('raspicam2'),
                                        'cfg',
                                        'params.yaml'))

    return LaunchDescription([
        DeclareLaunchArgument(
            name='param_dir',
            default_value=param_dir,
            description='camera param file path'),
        IncludeLaunchDescription(
            PathJoinSubstitution([
                FindPackageShare('turtlebot3_bringup'),
                'launch',
                'robot.launch.py']),
            ),
        Node(package="raspicam2",
             executable="raspicam2_node",
             parameters=[param_dir]),
        ])
