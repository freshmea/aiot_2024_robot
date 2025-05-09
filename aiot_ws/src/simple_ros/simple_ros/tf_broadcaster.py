import rclpy
import rclpy.logging
from geometry_msgs.msg import TransformStamped
from numpy import pi
from rclpy.node import Node
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf_transformations import quaternion_from_euler


class StaticFramePublisher(Node):
    def __init__(self):
        super().__init__('static_turtle_tf2_broadcaster')

        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.make_transforms()

    def make_transforms(self):
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'map'

        t.transform.translation.x = 1.0
        t.transform.translation.y = 1.0
        t.transform.translation.z = 0.0
        quat = quaternion_from_euler(pi/2, 0.0, pi/2)
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.tf_static_broadcaster.sendTransform(t)


def main():
    rclpy.init()
    node = StaticFramePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == "__main__":
    main()
