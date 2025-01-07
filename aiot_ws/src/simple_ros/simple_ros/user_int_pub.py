import rclpy
from rclpy.node import Node
from user_interface.msg import UserInt


class User_int_pub(Node):
    def __init__(self):
        super().__init__("user_int_pub")
        self.create_timer(1, self.print_hello)
        self.pub = self.create_publisher(UserInt, "send", 10)
        self.number = 0

    def print_hello(self):
        msg = UserInt()
        msg.stamp = self.get_clock().now().to_msg()
        msg.user_int = 5
        msg.user_int2 = 6
        msg.user_int3 = 7
        self.pub.publish(msg)

def main():
    rclpy.init()
    node = User_int_pub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == "__main__":
    main()
