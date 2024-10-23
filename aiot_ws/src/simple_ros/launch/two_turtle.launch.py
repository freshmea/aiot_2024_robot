import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    param_dir = LaunchConfiguration('param_dir',
                                    default=os.path.join(
                                        get_package_share_directory('simple_ros'),
                                        'param',
                                        'turtlesim.yaml'))
    return LaunchDescription([
        DeclareLaunchArgument('param_dir',
                              default_value=param_dir,
                              description="simple parameter"),
        Node(package="turtlesim",
             executable="turtlesim_node",
             parameters=[param_dir]
             ),
        ExecuteProcess(cmd=[['ros2 service call ', '/spawn ', 'turtlesim/srv/Spawn ', '"{x: 2, y: 2, theta: 0.2}"' ]], shell=True),
        Node(package="simple_ros",
             executable="move_turtle",
             parameters=[param_dir]
             ),
        

        ])
