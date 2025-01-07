import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from std_msgs.msg import Header


class PublishMap(Node):
    def __init__(self):
        super().__init__('publish_map')
        self.pub = self.create_publisher(OccupancyGrid, 'map', 10)
        self.timer = self.create_timer(0.001, self.pub_callback)  # 1ms timer

        # Map info
        self.msg = OccupancyGrid()
        self.msg.info.resolution = 0.1
        self.msg.info.width = 100
        self.msg.info.height = 100
        self.msg.info.origin.position.x = -(self.msg.info.width * self.msg.info.resolution) / 2.0
        self.msg.info.origin.position.y = -(self.msg.info.height * self.msg.info.resolution) / 2.0
        self.msg.info.origin.position.z = 0.0
        self.msg.info.origin.orientation.x = 0.0
        self.msg.info.origin.orientation.y = 0.0
        self.msg.info.origin.orientation.z = 0.0
        self.msg.info.origin.orientation.w = 1.0

        self.msg.data = [-1 for _ in range(self.msg.info.width * self.msg.info.height)]

        self.count = 0
        self.row = 0

    def pub_callback(self):
        self.msg.header = Header()
        self.msg.header.frame_id = 'odom'
        self.msg.header.stamp = self.get_clock().now().to_msg()

        # Update map data
        index = self.count + (self.msg.info.width * self.row)
        if self.msg.data[index] == -1:
            self.msg.data[index] = 100
        else:
            self.msg.data[index] = -1

        self.count += 1
        if self.count >= self.msg.info.width:
            self.count = 0
            self.row += 1

        if self.row >= self.msg.info.height:
            self.row = 0

        # Publish the map
        self.pub.publish(self.msg)

def main(args=None):
    rclpy.init(args=args)
    node = PublishMap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
