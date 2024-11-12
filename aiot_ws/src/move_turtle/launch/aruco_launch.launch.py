import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    info_topic = LaunchConfiguration('info_topic',
                                     default='/camera/image/camera_info')
    size = LaunchConfiguration('size', default=0.1)
    dictionary = LaunchConfiguration('dictionary', default='DICT_5X5_100')
    in_compressed = LaunchConfiguration('in_compressed',
                                        default='/camera/image/compressed')
    return LaunchDescription([
        DeclareLaunchArgument('info_toppic',
                              default_value=info_topic,
                              description="info_topic"),
        DeclareLaunchArgument('size',
                              default_value=size,
                              description="marker_size"),
        DeclareLaunchArgument('dictionary',
                              default_value=dictionary,
                              description="aruco dictionary id"),
        DeclareLaunchArgument('in_compressed',
                              default_value=in_compressed,
                              description="compressed topic name"),
        Node(package="ros2_aruco",
             executable="aruco_node",
             parameters=[{'camera_info_topic': info_topic,
                          'marker_size': size,
                          'aruco_dictionary_id': dictionary}]),
        Node(package="image_trsnsport",
             executable="republish",
             arguments=['compressed', 'raw'],
             remappings=[
                 ('in/compressed', in_compressed),
                 ('out', '/camera/image_raw')]),
        ]),
