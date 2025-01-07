import rclpy
from rclpy.node import Node
from rclpy.qos import (  # qos_profile_sensor_data,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import String


class Hello_sub(Node):
    def __init__(self):
        super().__init__("hellosub")
        self.qos_profile = QoSProfile(history=QoSHistoryPolicy.KEEP_ALL,
                                      reliability=QoSReliabilityPolicy.RELIABLE,
                                      durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        # self.qos_profile = qos_profile_sensor_data
        self.create_subscription(String, "send", self.sub_callback, self.qos_profile)

    def sub_callback(self, msg: String):
        self.get_logger().info(msg.data)

def main():
    rclpy.init()
    node = Hello_sub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == "__main__":
    main()
