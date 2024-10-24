import rclpy
from rclpy.node import Node
from user_interface.msg import ArithmeticArgument


class Argument(Node):
    def __init__(self):
        super().__init__("argument")
        self.create_timer(1.0, self.pub)
        self.pub_o = self.create_publisher(ArithmeticArgument, "arithmetic_argument", 10)

    def pub(self):
        msg = ArithmeticArgument()
        msg.argument_a = 4.0
        msg.argument_b = 9.0
        self.pub_o.publish(msg)

def main():
    rclpy.init()
    node = Argument()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()

if __name__ == "__main__":
    main()
