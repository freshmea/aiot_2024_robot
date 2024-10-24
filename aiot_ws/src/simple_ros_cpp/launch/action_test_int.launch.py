from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    step = LaunchConfiguration('step', default='5')
    return LaunchDescription([
        Node(package="simple_ros_cpp", executable="action_server"),
        Node(package="simple_ros_cpp", executable="action_client", arguments=[step])
        ])
