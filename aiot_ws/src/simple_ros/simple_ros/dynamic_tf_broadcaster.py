import rclpy
import rclpy.logging
from geometry_msgs.msg import TransformStamped
from numpy import cos, pi, sin
from rclpy.node import Node
from tf2_ros.transform_broadcaster import TransformBroadcaster
from tf_transformations import quaternion_from_euler


class DynamicFramePublisher(Node):
    def __init__(self):
        super().__init__('dynamic_tf2_broadcaster')

        self.tf_static_broadcaster = TransformBroadcaster(self)
        self.create_timer(0.1, self.timer_callback)
        self.t = 0.0

    def timer_callback(self):
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'map'

        t.transform.translation.x = 1.0 * cos(self.t)
        t.transform.translation.y = 1.0 * sin(self.t)
        t.transform.translation.z = 0.0
        quat = quaternion_from_euler(pi/2, 0.0, pi/2)
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        self.t += 0.01

        self.tf_static_broadcaster.sendTransform(t)


def main():
    rclpy.init()
    node = DynamicFramePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == "__main__":
    main()
