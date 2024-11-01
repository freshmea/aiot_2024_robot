from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    container = ComposableNodeContainer(
        name='mycontainer',
        namespace='',
        package='simple_ros_cpp',
        composable_node_descriptions=[
            ComposableNode(
                package='simple_ros_cpp',
                plugin='composition::Talker',
                name='talker'),
            ComposableNode(
                package='simple_ros_cpp',
                plugin='composition::Listener',
                name='listener')
        ],
        output='screen',
        )
    return LaunchDescription([container])
