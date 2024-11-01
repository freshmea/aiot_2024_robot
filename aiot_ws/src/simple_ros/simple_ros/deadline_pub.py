import random
import time

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos_event import PublisherEventCallbacks
from std_msgs.msg import String


class Deadline_pub(Node):
    def __init__(self):
        super().__init__("hello_pub")
        publisher_callbacks = PublisherEventCallbacks(
        deadline = self.deadline_callback)
        deadline = Duration(seconds=6)
        self.qos_profile = QoSProfile(depth=10, deadline=deadline)
        self.create_timer(4, self.print_hello)
        self.pub = self.create_publisher(String, "send", self.qos_profile, event_callbacks=publisher_callbacks)
        self.number = 0

    def print_hello(self):
        msg = String()
        msg.data = f"hello, ros2! nice to meet you! + {self.number}"
        tm = random.random()*4
        self.get_logger().info(f"{tm}")
        time.sleep(tm)
        self.pub.publish(msg)
        self.get_logger().info(msg.data)
        self.number += 1

    def deadline_callback(self, event):
        self.get_logger().info(f"task over dealine ----{self.number}")
        self.get_logger().info(f"total count : {event.total_count}")
        self.get_logger().info(f"total count change : {event.total_count_change}")

def main():
    rclpy.init()
    node = Deadline_pub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()

if __name__ == "__main__":
    main()
