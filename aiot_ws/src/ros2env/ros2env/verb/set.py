import os

from ros2env.verb import VerbExtension


class SetVerb(VerbExtension):
    def add_arguments(self, parser, cli_name):
        parser.add_argument('env_name', help='Name of env')
        parser.add_argument('value', help='value of env')

    def main(self, *, args):
        if args.env_name or args.value:
            print('test')
            os.environ[args.env_name] = args.value
            print(os.getenv(args.env_name))
