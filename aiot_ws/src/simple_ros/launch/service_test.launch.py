from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package="simple_ros", executable="service_server"),
        Node(package="simple_ros", executable="service_client")
        ])
