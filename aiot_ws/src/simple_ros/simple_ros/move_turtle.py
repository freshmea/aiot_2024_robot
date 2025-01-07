import rclpy
from geometry_msgs.msg import Twist
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

    def twist_pub(self):
        self.pub.publish(self.twist)

    def pose_callback(self, msg: Pose):
        self.pose = msg

    def color_callback(self, msg: Color):
        self.color = msg
    
    def update(self):
        """ self.twist, self.pose, self.color 을 이용한 알고리즘"""
        # self.twist.linear.x += 0.001
        # self.twist.angular.z = 1.0
        if self.phase == 0:
            if self.pose.theta < 0:
                self.twist.linear.x = 0.0
                self.twist.angular.z = 1.0
            elif self.pose.x < 8:
                self.twist.linear.x = 0.6
                self.twist.angular.z = 0.0
            else:
                self.phase = 1
        elif self.phase == 1:
            if self.pose.theta < 3.141592/2:
                self.twist.linear.x = 0.0
                self.twist.angular.z = 1.0
            elif self.pose.y < 8:
                self.twist.linear.x = 0.6
                self.twist.angular.z = 0.0
            else:
                self.phase = 2
        elif self.phase == 2:
            if self.pose.theta < 3.0:
                self.twist.linear.x = 0.0
                self.twist.angular.z = 1.0
            elif self.pose.x > 2:
                self.twist.linear.x = 0.6
                self.twist.angular.z = 0.0
            else:
                self.phase = 3
                self.get_logger().info("phase 3")
        elif self.phase == 3:
            if  not -3.341592/2 < self.pose.theta < -3.141592/2:
                self.twist.linear.x = 0.0
                self.twist.angular.z = 1.0
            elif self.pose.y > 2:
                self.twist.linear.x = 0.6
                self.twist.angular.z = 0.0
            else:
                self.phase = 0

def main():
    rclpy.init()
    node = Move_turtle()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == "__main__":
    main()
