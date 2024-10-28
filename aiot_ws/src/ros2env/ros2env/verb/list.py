import os

from ros2env.verb import VerbExtension


def get_ros_env_list():
    ros_env_list = f"ROS_VERSION \t: {os.getenv('ROS_VERSION', 'None')}\n"
    ros_env_list += f"ROS_DISTRO \t: {os.getenv('ROS_DISTRO', 'None')}\n"
    ros_env_list += f"ROS_PYTHON_V \t: {os.getenv('ROS_PYTHON_VERSION', 'None')}\n"
    return ros_env_list

def get_dds_env_list():
    ros_env_list = f"ROS_DOMAIN_ID \t: {os.getenv('ROS_DOMAIN_ID', 'None')}\n"
    ros_env_list += f"DDS VENDOR \t: {os.getenv('RMW_IMPLEMENTATION', 'None')}\n"
    return ros_env_list

def get_all_env_list():
    ros_env_list = get_ros_env_list()
    dds_env_list = get_dds_env_list()
    all1 = ros_env_list + dds_env_list
    return all1


class ListVerb(VerbExtension):

    def add_arguments(self, parser, cli_name):
        parser.add_argument('-a', '--all', action ="store_true", help='display')
        parser.add_argument('-r', '--ros_env', action ="store_true", help='display')
        parser.add_argument('-d', '--dds_env', action ="store_true", help='display')

    def main(self, *, args):
        if args.ros_env:
            message = get_ros_env_list()
        elif args.dds_env:
            message = get_dds_env_list()
        else:
            message = get_all_env_list()
        print(message)

