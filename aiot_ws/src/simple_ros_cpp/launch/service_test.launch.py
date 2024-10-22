from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package="simple_ros_cpp", executable="service_server"),
        Node(package="simple_ros_cpp", executable="service_client")
        ])
