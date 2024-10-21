import rclpy
from rclpy.node import Node
from rclpy.qos import (  # qos_profile_sensor_data,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import Header, String


class Message_time_sub(Node):
    def __init__(self):
        super().__init__("mtsub")
        self.qos_profile = QoSProfile(history=QoSHistoryPolicy.KEEP_ALL,
                                      reliability=QoSReliabilityPolicy.RELIABLE,
                                      durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "message", self.sub_callback, self.qos_profile)
        self.create_subscription(Header, "time", self.sub_callback2, self.qos_profile)

    def sub_callback(self, msg: String):
        print(msg.data)

    def sub_callback2(self, msg: Header):
        print(msg.frame_id, msg.stamp.sec,  msg.stamp.nanosec)

def main():
    rclpy.init()
    node = Message_time_sub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()

if __name__ == "__main__":
    main()
