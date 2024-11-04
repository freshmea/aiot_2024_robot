from launch import LaunchDescription
from launch_ros.actions import LifecycleNode, Node


def generate_launch_description():
    return LaunchDescription([
        LifecycleNode(
            package="simple_ros_cpp",
            executable="lifecycle_talker",
            name="lc_talker",
            namespace="",
            respawn=True,
            respawn_delay=2.0),
        Node(package="simple_ros_cpp", executable="lifecycle_listener")
        ])
