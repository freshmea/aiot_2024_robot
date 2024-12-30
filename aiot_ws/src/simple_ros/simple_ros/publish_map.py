import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from std_msgs.msg import Header


class PublishMap(Node):
    def __init__(self):
        super().__init__('publish_map')
        self._pub = self.create_publisher(OccupancyGrid, 'map', 1)
        self._timer = self.create_timer(0.001, self.pub_callback)  # 1ms timer

        # Map info
        self._msg = OccupancyGrid()
        self._msg.info.resolution = 0.1
        self._msg.info.width = 100
        self._msg.info.height = 100
        self._msg.info.origin.position.x = -(self._msg.info.width * self._msg.info.resolution) / 2.0
        self._msg.info.origin.position.y = -(self._msg.info.height * self._msg.info.resolution) / 2.0
        self._msg.info.origin.position.z = 0.0
        self._msg.info.origin.orientation.x = 0.0
        self._msg.info.origin.orientation.y = 0.0
        self._msg.info.origin.orientation.z = 0.0
        self._msg.info.origin.orientation.w = 1.0

        self._msg.data = [-1 for _ in range(self._msg.info.width * self._msg.info.height)]

        self._count = 0
        self._row = 0

    def pub_callback(self):
        self._msg.header = Header()
        self._msg.header.frame_id = 'odom'
        self._msg.header.stamp = self.get_clock().now().to_msg()

        # Update map data
        index = self._count + (self._msg.info.width * self._row)
        if self._msg.data[index] == -1:
            self._msg.data[index] = 100
        else:
            self._msg.data[index] = -1

        self._count += 1
        if self._count >= self._msg.info.width:
            self._count = 0
            self._row += 1

        if self._row >= self._msg.info.height:
            self._count = 0
            self._row = 0


        # Publish the map
        self._pub.publish(self._msg)


def main(args=None):
    rclpy.init(args=args)
    node = PublishMap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
