import math

import rclpy
import serial
import tf_transformations
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from std_msgs.msg import String
from tf2_ros import Buffer, TransformBroadcaster


class Arduino_servo(Node):
    def __init__(self):
        super().__init__("arduino_servo")
        self.create_subscription(String, "servo", self.sub_callback, 10)
        self.ser = serial.Serial('/dev/ttyACM0', 115200)
        self.tf_broadcaster = TransformBroadcaster(self)

    def sub_callback(self, msg: String):
        servo_tf = TransformStamped()
        servo_tf.header.stamp = self.get_clock().now().to_msg()
        servo_tf.header.frame_id = "base_link"
        servo_tf.child_frame_id = "flag"
        theta = int(msg.data[4:7])
        q = tf_transformations.quaternion_from_euler(0, -theta/180*math.pi, 0)
        servo_tf.transform.translation.x = 0.025
        servo_tf.transform.translation.y = -0.05
        servo_tf.transform.translation.z = 0.14
        servo_tf.transform.rotation.x = q[0]
        servo_tf.transform.rotation.y = q[1]
        servo_tf.transform.rotation.z = q[2]
        servo_tf.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(servo_tf)

        byte_msg = (msg.data+'\n').encode('utf-8')
        self.ser.write(byte_msg)
        self.get_logger().info(msg.data)

def main():
    rclpy.init()
    node = Arduino_servo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == "__main__":
    main()
