from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package="simple_ros", executable="hello_pub"),
        Node(package="simple_ros", executable="hello_sub")
        ])
