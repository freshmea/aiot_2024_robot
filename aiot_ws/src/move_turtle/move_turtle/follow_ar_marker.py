import math

import rclpy
import tf2_ros
import tf_transformations
from geometry_msgs.msg import Pose, TransformStamped, Twist
from nav_msgs.msg import Odometry

# Duration
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from ros2_aruco_interfaces.msg import ArucoMarkers
from sensor_msgs.msg import BatteryState, Imu, LaserScan
from tf2_ros import Buffer, TransformBroadcaster, TransformListener

MAX_VEL = 0.21
MAX_ANGLE = 2.8 # radian/sec


class Move_turtle(Node):
    def __init__(self):
        super().__init__("follow_ar_marker")
        self.qos_profile = qos_profile_sensor_data
        self.create_timer(0.1, self.twist_pub)
        self.create_timer(1/60, self.update)
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(LaserScan, "/scan", self.laser_callback, self.qos_profile)
        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.create_subscription(Imu, "/imu", self.imu_callback, 10)
        self.create_subscription(BatteryState, "/battery_state", self.battery_callback, 10)
        self.create_subscription(ArucoMarkers, "/aruco_markers", self.aruco_callback, 10)
        self.twist = Twist()
        self.laserscan = LaserScan()
        self.odom = Odometry()
        self.imu = Imu()
        self.battery = BatteryState()
        self.aruco_markers = ArucoMarkers()
        self.follow_tf = Pose()
        self.follow_tf2 = TransformStamped()
        self.theta = 0.0 # raian
        self.phase = 0
        self.laserscan_degree = [3.5 for i in range(360)]
        self.find_wall = False
        self.tf_broadcaster = TransformBroadcaster(self)

    def twist_pub(self):
        self.restrain()
        self.pub.publish(self.twist)

    def laser_callback(self, msg: LaserScan):
        self.laserscan = msg
        count = 0
        for s_radian in self.laserscan.ranges:
            radian_index = msg.angle_min+msg.angle_increment*count
            degree_index = int(radian_index/3.141592*180)
            if s_radian == float('inf') or s_radian == 0.0:
                s_radian = msg.range_max
            self.laserscan_degree[degree_index] = s_radian
            count +=1
    
    def aruco_callback(self, msg: ArucoMarkers):
        self.aruco_markers = msg
        for marker_id_ele in msg.marker_ids:
            if marker_id_ele == 1:
                self.follow_tf = msg.poses[msg.marker_ids.index(marker_id_ele)]
        self.aruco_tf_publish_function()

    def aruco_tf_publish_function(self):
        # tf2로 구현
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "camera_link_optical"
        t.child_frame_id = "follow_point"
        t.transform.translation.x = self.follow_tf.position.x
        t.transform.translation.y = self.follow_tf.position.y
        t.transform.translation.z = self.follow_tf.position.z
        t.transform.rotation.x = self.follow_tf.orientation.x
        t.transform.rotation.y = self.follow_tf.orientation.y
        t.transform.rotation.z = self.follow_tf.orientation.z
        t.transform.rotation.w = self.follow_tf.orientation.w
        self.tf_broadcaster.sendTransform(t)


    def odom_callback(self, msg: Odometry):
        self.odom = msg
        x = msg.pose.pose.orientation.x
        y = msg.pose.pose.orientation.y
        z = msg.pose.pose.orientation.z
        w = msg.pose.pose.orientation.w
        _, _, self.theta = tf_transformations.euler_from_quaternion((x, y, z, w))


    def imu_callback(self, msg: Imu):
        self.imu = msg
        # self.get_logger().info(f"IMU : {msg.orientation.x}")

    def battery_callback(self, msg: BatteryState):
        self.battery = msg
        # self.get_logger().info(f"battery : {msg.percentage}")

    def update(self):
        """ self.twist, self.pose, self.color 을 이용한 알고리즘"""
        buffer = Buffer()
        self.tf_listener = TransformListener(buffer, self)
        try:
            self.follow_tf2 = buffer.lookup_transform("camera_link", "follow_point", tf2_ros.Time())
            self.get_logger().info(f"follow_tf : {self.follow_tf2}")
            self.twist.angular.z = math.atan2(
                self.follow_tf2.transform.translation.y,
                self.follow_tf2.transform.translation.x)
            self.twist.linear.x = math.sqrt(
                self.follow_tf2.transform.translation.x**2 +
                self.follow_tf2.transform.translation.y**2)
        except Exception as e:
            self.get_logger().info(f"Exception : {e}")
            self.twist.angular.z = math.atan2(
                self.follow_tf2.transform.translation.y,
                self.follow_tf2.transform.translation.x)
            self.twist.linear.x = math.sqrt(
                self.follow_tf2.transform.translation.x**2 +
                self.follow_tf2.transform.translation.y**2)

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
