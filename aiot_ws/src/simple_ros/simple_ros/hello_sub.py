import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import String


class Hello_sub(Node):
    def __init__(self):
        super().__init__("hellosub")
        self.qos_profile = QoSProfile()
        self.qos_profile.history = QoSHistoryPolicy.KEEP_ALL
        self.qos_profile.history = QoSReliabilityPolicy.RELIABLE
        self.qos_profile.history = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(String, "send", self.sub_callback, self.qos_profile)

    def sub_callback(self, msg: String):
        print(msg.data)

def main():
    rclpy.init()
    node = Hello_sub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()

if __name__ == "__main__":
    main()
