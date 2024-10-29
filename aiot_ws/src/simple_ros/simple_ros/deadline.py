import argparse
import sys

import rclpy
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.logging import get_logger
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos_event import PublisherEventCallbacks, SubscriptionEventCallbacks
from std_msgs.msg import String


class Talker(Node):
    def __init__(self, topic_name, qos_profile, event_callbacks, publish_count=0, assert_topic_period=None):
        super().__init__('qos_talker')
        self.publisher = self.create_publisher(String, topic_name, qos_profile, event_callbacks=event_callbacks)
        self.pub_timer = self.create_timer(0.5, self.publish)
        if assert_topic_period:
            self.assert_topic_timer = self.create_timer(assert_topic_period, self.publisher.assert_liveliness)
        else:
            self.assert_topic_timer= None
        self.pause_timer = None
        self.publish_count = 0
        self.stop_at_count = publish_count

    def publish(self):
        msg = String()
        msg.data = f'Talker says {self.publish_count}'
        self.publish_count += 1
        if self.stop_at_count > 0 and self.publish_count >= self.stop_at_count:
            self.pub_timer.cancel()
        self.publisher.publish(msg)

class Listener(Node):
    def __init__(self, topic_name, qos_profile, event_callbacks, defer_subscribe=False):
        super().__init__('qos_listener')
        self.subscriptoin = None
        self.topic_name = topic_name
        self.qos_profile = qos_profile
        self.event_callbacks = event_callbacks
        if not defer_subscribe:
            self.start_listening()

    def start_listening(self):
        if not self.subscriptoin:
            self.sub = self.create_subscription(
                String, self.topic_name, self._message_callback,
                self.qos_profile,
                event_callbacks = self.event_callbacks
            )

    def _message_callback(self, message):
        self.get_logger().info(f"I heard: {message.data}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('deadline', type=int, help='deadline QoS setting milliseconds')
    parser.add_argument('--publish-for', type=int, dfault=5000, help='publish until pausing the talker')
    parser.add_argument('--pause-for', type=int, default=1000, help='pause the talker before beginning publish again')
    return parser.parse_args()

def main(args=None):
    parsed_args = parse_args()
    rclpy.init(args=args)
    topic = 'qos_deadline_chatter'
    deadline = Duration(seconds=parsed_args.deadline/1000.0)
    qos_profile = QoSProfile(depth=10, deadline=deadline)
    subscription_callbacks = SubscriptionEventCallbacks(
        deadline = lambda event: get_logger("Listener").info(str(event))) # type : ignore
    listener = Listener(topic, qos_profile, event_callbacks=subscription_callbacks)

if __name__ == '__main__':
    main()
