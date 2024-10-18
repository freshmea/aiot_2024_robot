import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import Header


class Time_pub(Node):
    def __init__(self):
        super().__init__("time_pub")
        self.qos_profile = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST,
                                      reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                      durability=QoSDurabilityPolicy.VOLATILE,
                                      depth=10)
        self.create_timer(1, self.print_hello)
        self.pub = self.create_publisher(Header, "time", self.qos_profile)
        self.clock = self.get_clock()

    def print_hello(self):
        msg = Header()
        msg.frame_id = "time"
        msg.stamp = self.clock.now().to_msg()
        print(f"sec: {msg.stamp.sec}, nano sec : {msg.stamp.nanosec}")
        self.pub.publish(msg)

def main():
    rclpy.init()
    node = Time_pub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()

if __name__ == "__main__":
    main()
