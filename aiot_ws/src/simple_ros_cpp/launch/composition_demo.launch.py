from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    container = ComposableNodeContainer(
        name='mycontainer',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='simple_ros_cpp',
                plugin='composition::Talker',
                name='talker'),
            ComposableNode(
                package='simple_ros_cpp',
                plugin='composition::Listener',
                name='listener'),
            ComposableNode(
                package='composition',
                plugin='composition::Server',
                name='server'),
            ComposableNode(
                package='composition',
                plugin='composition::Client',
                name='client')
        ],
        output='screen',
        )
    return LaunchDescription([container])
