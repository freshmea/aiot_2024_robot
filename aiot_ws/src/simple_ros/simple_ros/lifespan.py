import argparse
import sys

import rclpy
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.logging import get_logger
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.qos_event import PublisherEventCallbacks, SubscriptionEventCallbacks
from simple_ros.deadline import Listener, Talker
from std_msgs.msg import String


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('lifespan', type=int, help='lifespan QoS setting milliseconds')
    parser.add_argument('--history', type=int, default=10, help='history depth QoS setting milliseconds')
    parser.add_argument('--publish-count', type=int, default=10, help='how may message to publish before stopping')
    parser.add_argument('--subscribe-after', type=int, default=2500, help='sub create after startup')
    return parser.parse_args()

def main(args=None):
    parsed_args = parse_args()
    rclpy.init(args=args)
    topic = 'qos_lifespan_chatter'
    lifespan = Duration(seconds=parsed_args.lifespan/1000.0)
    qos_profile = QoSProfile(depth=10,
                             reliability=QoSReliabilityPolicy.RELIABLE,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                             lifespan=lifespan)

    listener = Listener(topic, qos_profile, event_callbacks=None, defer_subscribe=True)
    talker = Talker(topic, qos_profile, event_callbacks=None, publish_count=parsed_args.publish_count)
    subscribe_timer = listener.create_timer(parsed_args.subscribe_after/1000.0, lambda: listener.start_listening())

    publish_for_seconds = parsed_args.publish_for / 1000.0
    pause_for_seconds = parsed_args.pause_for / 1000.0
    pause_timer = talker.create_timer(publish_for_seconds, lambda: talker.pause_for(pause_for_seconds))

    executor = SingleThreadedExecutor()
    executor.add_node(listener)
    executor.add_node(talker)
    try:
        executor.spin()
    except KeyboardInterrupt:
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
