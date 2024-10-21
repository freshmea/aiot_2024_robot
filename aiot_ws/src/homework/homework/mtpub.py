import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import Header, String


class Message_time_pub(Node):
    def __init__(self):
        super().__init__("mtpub")
        self.qos_profile = QoSProfile(history=QoSHistoryPolicy.KEEP_ALL,
                                      reliability=QoSReliabilityPolicy.RELIABLE,
                                      durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_timer(1, self.print_message)
        self.create_timer(0.5, self.print_time)
        self.pub = self.create_publisher(String, "message2", self.qos_profile)
        self.pub2 = self.create_publisher(Header, "time", self.qos_profile)
        self.number = 0

    def print_message(self):
        msg = String()
        msg.data = f"this is message from mtpub! + {self.number}"
        self.pub.publish(msg)
        print(msg.data)
        self.number += 1

    def print_time(self):
        msg = Header()
        msg.frame_id = "time"
        msg.stamp = self.get_clock().now().to_msg()
        self.pub2.publish(msg)
        print(msg)

def main():
    rclpy.init()
    node = Message_time_pub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()

if __name__ == "__main__":
    main()
