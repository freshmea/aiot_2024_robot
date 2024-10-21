import rclpy
from rclpy.node import Node
from rclpy.qos import (  # qos_profile_sensor_data,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import String


class Message_sub(Node):
    def __init__(self):
        super().__init__("msub")
        self.qos_profile = QoSProfile(history=QoSHistoryPolicy.KEEP_ALL,
                                      reliability=QoSReliabilityPolicy.RELIABLE,
                                      durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "message", self.sub_callback, self.qos_profile)

    def sub_callback(self, msg: String):
        print(msg.data)

def main():
    rclpy.init()
    node = Message_sub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()

if __name__ == "__main__":
    main()
