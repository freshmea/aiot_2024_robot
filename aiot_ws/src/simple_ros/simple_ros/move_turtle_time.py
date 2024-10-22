import rclpy
from geometry_msgs.msg import Twist
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from turtlesim.msg import Color, Pose


class Move_turtle(Node):
    def __init__(self):
        super().__init__("hello_pub")
        self.qos_profile = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST,
                                      reliability=QoSReliabilityPolicy.RELIABLE,
                                      durability=QoSDurabilityPolicy.VOLATILE,
                                      depth=10)
        self.create_timer(0.1, self.twist_pub)
        self.create_timer(1/60, self.update)
        self.pub = self.create_publisher(Twist, "turtle1/cmd_vel", self.qos_profile)
        self.create_subscription(Pose, "turtle1/pose", self.pose_callback, self.qos_profile)
        self.create_subscription(Color, "turtle1/color_sensor", self.color_callback, self.qos_profile)
        self.twist = Twist()
        self.pose = Pose()
        self.color = Color()
        self.phase = 0
        self.prevTime = self.get_clock().now()

    def twist_pub(self):
        self.pub.publish(self.twist)

    def pose_callback(self, msg: Pose):
        self.pose = msg

    def color_callback(self, msg: Color):
        self.color = msg

    def update(self):
        """ self.twist, self.pose, self.color 을 이용한 알고리즘"""
        if self.phase == 0:
            self.twist.linear.x = 0.0
            self.twist.angular.z = 2.0
            if (self.get_clock().now() - self.prevTime) > Duration(seconds=1, nanoseconds=250_000_000):
                self.prevTime = self.get_clock().now()
                self.phase = 1
        elif self.phase == 1:
            self.twist.linear.x = 1.0
            self.twist.angular.z = 0.0
            if (self.get_clock().now() - self.prevTime) > Duration(seconds=2):
                self.prevTime = self.get_clock().now()
                self.phase = 0

def main():
    rclpy.init()
    node = Move_turtle()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()

if __name__ == "__main__":
    main()
