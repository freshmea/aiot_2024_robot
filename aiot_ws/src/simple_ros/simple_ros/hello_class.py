import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class Hello(Node):
    def __init__(self):
        super().__init__("hello")
        self.create_timer(1, self.print_hello)
        self.pub = self.create_publisher(String, "send", 10)
        self.number = 0

    def print_hello(self):
        msg = String()
        msg.data = f"hello, ros2! nice to meet you! + {self.number}"
        self.pub.publish(msg)
        print("hello, ros2! nice to meet you!")
        self.number += 1

def main():
    rclpy.init()
    node = Hello()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == "__main__":
    main()
