import os

from ros2env.verb import VerbExtension


def get_ros_env_list():
    ros_env_list = f"ROS_VERSION : {os.getenv('ROS_VERSION', 'None')}"
    return ros_env_list

class ListVerb(VerbExtension):

    def add_arguments(self, parser, cli_name):
        parser.add_argument('-a', '--all', action ="store_true", help='display')

    def main(self, *, args):
        # message = None
        # if args.all:
        message = get_ros_env_list()
        print(message)

