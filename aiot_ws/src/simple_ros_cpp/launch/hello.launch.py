from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package="simple_ros_cpp", executable="hello_pub_class"),
        Node(package="simple_ros_cpp", executable="hello_sub")
        ])
