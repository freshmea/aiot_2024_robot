from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_rviz_config_path = PathJoinSubstitution([
                FindPackageShare('move_turtle'),
                'rviz',
                'urdf.rviz'])
    default_model_path = PathJoinSubstitution([
                'urdf',
                'myfirst.urdf'])
    model = DeclareLaunchArgument(
        name='model',
        default_value=default_model_path,
        description='urdf path')
    gui_arg = DeclareLaunchArgument(
        name='gui',
        default_value='true',
        choices=['true', 'false'],
        description='Flag to enable joint_state_publisher_gui')
    rviz_arg = DeclareLaunchArgument(
        name='rvizconfig',
        default_value=default_rviz_config_path,
        description='Absolute path to rviz config file')
    return LaunchDescription([
        model,
        gui_arg,
        rviz_arg,
        IncludeLaunchDescription(
            PathJoinSubstitution([
                FindPackageShare('urdf_launch'),
                'launch',
                'display.launch.py']),
            launch_arguments={
                'urdf_package': 'urdf_tutorial',
                'urdf_package_path': LaunchConfiguration('model'),
                'rviz_config': LaunchConfiguration('rvizconfig'),
                'jsp_gui': LaunchConfiguration('gui')
            }.items()
            )])
