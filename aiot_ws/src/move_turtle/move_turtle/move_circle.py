import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import BatteryState, Imu, LaserScan
from tf_transformations import euler_from_quaternion

MAX_VEL = 0.21
MAX_ANGLE = 2.8 # radian/sec


class Move_turtle(Node):
    def __init__(self):
        super().__init__("hello_pub")
        self.qos_profile = qos_profile_sensor_data
        self.create_timer(0.1, self.twist_pub)
        self.create_timer(1/60, self.update)
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(LaserScan, "/scan", self.laser_callback, self.qos_profile)
        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.create_subscription(Imu, "/imu", self.imu_callback, 10)
        self.create_subscription(BatteryState, "/battery_state", self.battery_callback, 10)
        self.twist = Twist()
        self.laserscan = LaserScan()
        self.odom = Odometry()
        self.imu = Imu()
        self.battery = BatteryState()
        self.theta = 0.0 # raian
        self.phase = 0

    def twist_pub(self):
        self.restrain()
        self.pub.publish(self.twist)

    def laser_callback(self, msg: LaserScan):
        self.laserscan = msg
        self.get_logger().info(f"laserscan : {msg.ranges[0]}")

    def odom_callback(self, msg: Odometry):
        self.odom = msg
        _, _, self.theta = euler_from_quaternion(msg.pose.pose.orientation)
        self.get_logger().info(f"odom yaw(theta): {self.theta}")

    def imu_callback(self, msg: Imu):
        self.imu = msg
        self.get_logger().info(f"IMU : {msg.orientation.x}")

    def battery_callback(self, msg: BatteryState):
        self.battery = msg
        self.get_logger().info(f"battery : {msg.percentage}")

    def update(self):
        """ self.twist, self.pose, self.color 을 이용한 알고리즘"""
        # self.twist.linear.x += 0.001
        # self.twist.angular.z = 1.0
        if self.phase == 0:
            if self.odom.pose.pose.position.x < 0.3 and -0.1 < self.theta < 0.1:
                self.twist.linear.x = 0.1  # Move forward
            elif self.odom.pose.pose.position.x < 0.3 and self.theta < -0.1:
                self.twist.angular.z = 0.3  # Adjust rotation
            elif self.odom.pose.pose.position.x < 0.3 and self.theta > 0.1:
                self.twist.angular.z = -0.3  # Adjust rotation
            elif self.theta < math.pi / 2:
                self.twist.angular.z = 0.3  # Turn 90 degrees
            else:
                self.phase += 1
        if self.phase == 1:
            if self.odom.pose.pose.position.y < 0.3 and 1.47 < self.theta < 1.67:
                self.twist.linear.x = 0.1
            elif self.odom.pose.pose.position.y < 0.3 and self.theta < 1.47:
                self.twist.angular.z = 0.3
            elif self.odom.pose.pose.position.y < 0.3 and self.theta > 1.67:
                self.twist.angular.z = -0.3
            elif 0 < self.theta < math.pi:
                self.twist.angular.z = 0.3
            else:
                self.theta += 1
        if self.phase == 2:
            if self.odom.pose.pose.position.x > 0 and self.theta > 3.04:
                self.twist.linear.x = 0.1
            elif self.odom.pose.pose.position.x > 0 and self.theta < -3.04:
                self.twist.linear.x = 0.1
            elif self.odom.pose.pose.position.x > 0 and self.theta < 3.04:
                self.twist.angular.z = 0.3
            elif self.odom.pose.pose.position.x > 0 and self.theta > -3.04:
                self.twist.angular.z = -0.3
            elif self.theta < -1.57:
                self.twist.angular.z = 0.3
            else:
                self.phase += 1
        if self.phase == 3:
            if self.odom.pose.pose.position.y > 0 and -1.67 < self.theta < -1.47:
                self.twist.linear.x = 0.1
            elif self.odom.pose.pose.position.y > 0 and self.theta < -1.67:
                self.twist.angular.z = 0.3
            elif self.odom.pose.pose.position.y > 0 and self.theta > -1.47:
                self.twist.angular.z = -0.3
            elif self.theta < 0:
                self.twist.angular.z = 0.3
            else:
                self.phase = 0 

    def restrain(self):
        self.twist.linear.x = min([self.twist.linear.x , MAX_VEL])
        self.twist.linear.x = max([self.twist.linear.x , -MAX_VEL])
        self.twist.angular.z = min([self.twist.angular.z , MAX_ANGLE])
        self.twist.angular.z = max([self.twist.angular.z , -MAX_ANGLE])

def main():
    rclpy.init()
    node = Move_turtle()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        for _ in range(10):
            node.pub.publish(Twist())
        node.destroy_node()

if __name__ == "__main__":
    main()
