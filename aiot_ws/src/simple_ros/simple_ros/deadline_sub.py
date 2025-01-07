import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos_event import SubscriptionEventCallbacks
from std_msgs.msg import String


class Deadline_sub(Node):
    def __init__(self):
        super().__init__("hello_sub")
        publisher_callbacks = SubscriptionEventCallbacks(
        deadline = self.deadline_callback)
        deadline = Duration(seconds=6)
        self.qos_profile = QoSProfile(depth=10, deadline=deadline)
        self.pub = self.create_subscription(String, "send", self.sub_callback,  self.qos_profile, event_callbacks=publisher_callbacks)
        self.prev_time = self.get_clock().now()

    def sub_callback(self, msg):
        self.get_logger().info(msg.data)
        print((self.get_clock().now() - self.prev_time).nanoseconds/1_000_000_000)
        self.prev_time = self.get_clock().now()

    def deadline_callback(self, event):
        self.get_logger().info("task over dealine ----")
        self.get_logger().info(f"total count : {event.total_count}")
        self.get_logger().info(f"total count change : {event.total_count_change}")

def main():
    rclpy.init()
    node = Deadline_sub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == "__main__":
    main()
