import argparse
import sys

import rclpy
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.logging import get_logger
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSLivelinessPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from rclpy.qos_event import PublisherEventCallbacks, SubscriptionEventCallbacks
from simple_ros.deadline import Listener, Talker
from std_msgs.msg import String


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('liveliness_lease_duration', type=int, help='live QoS setting milliseconds')
    parser.add_argument('--policy', type=str, default='auto', help='policy QoS setting milliseconds')
    parser.add_argument('--topic-assert-period', type=int, default=0, help='how often Talk live check')
    parser.add_argument('--kill-publisher-after', type=int, default=3000, help='shutdown Talker after this duration')
    return parser.parse_args()

def main(args=None):
    kill_timer =None
    parsed_args = parse_args()
    rclpy.init(args=args)
    topic = 'qos_liveliness_chatter'
    liveliness_lease_duration = Duration(seconds=parsed_args.liveliness_lease_duration/1000.0)


    # if parsed_args.policy == 'auto':
    #     policy = QoSLivelinessPolicy.AUTOMATIC
    # elif parsed_args.policy == 'manual':
    #     policy = QoSLivelinessPolicy.MANUAL_BY_TOPIC
    # else:
    #     policy = QoSLivelinessPolicy.AUTOMATIC

    POLICY_MAP= {
        'auto' : QoSLivelinessPolicy.AUTOMATIC,
        'manual' : QoSLivelinessPolicy.MANUAL_BY_TOPIC
    }
    policy = POLICY_MAP[parsed_args.policy]

    qos_profile = QoSProfile(depth=10,
                             liveliness=policy,
                             liveliness_lease_duration=liveliness_lease_duration)

    subscription_callbacks = SubscriptionEventCallbacks(
        liveliness=lambda event: get_logger('Listener').info(str(event)))
    listener = Listener(topic, qos_profile, event_callbacks=subscription_callbacks)
    
    publisher_callbacks = PublisherEventCallbacks(
        liveliness=lambda event: get_logger('Talker').info(str(event)))
    talker = Talker(topic, qos_profile, event_callbacks=publisher_callbacks,
                    assert_topic_period=parsed_args.topic_assert_period /1000.0)

    executor = SingleThreadedExecutor()
    def kill_talker():
        if policy == QoSLivelinessPolicy.AUTOMATIC:
            executor.remove_node(talker)
            talker.destroy_node()
        elif policy == QoSLivelinessPolicy.MANUAL_BY_TOPIC:
            talker.stop()
        kill_timer.cancel()

    if parsed_args.kill_publisher_after > 0:
        kill_timer = listener.create_timer(parsed_args.kill_publisher_after /1000.0, kill_talker)
        
    executor.add_node(listener)
    executor.add_node(talker)
    try:
        executor.spin()
    except KeyboardInterrupt:
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
