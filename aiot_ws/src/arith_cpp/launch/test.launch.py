from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package="arith_cpp", executable="argument"),
        Node(package="arith_cpp", executable="calculator")
        ])
