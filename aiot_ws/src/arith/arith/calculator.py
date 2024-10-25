import rclpy
from rclpy.node import Node
from user_interface.msg import ArithmeticArgument


class Calculator(Node):
    def __init__(self):
        super().__init__("calculator")
        self.create_subscription(ArithmeticArgument, "arithmetic_argument", self.sub_callback, 10)
        self.argument_a = 0
        self.argument_b = 0

    def sub_callback(self, msg: ArithmeticArgument):
        self.argument_a = msg.argument_a
        self.argument_b = msg.argument_b
        self.get_logger().info(f"Time Stamp : {msg.stamp}")
        self.get_logger().info(f"Argument_a : {self.argument_a}")
        self.get_logger().info(f"Argument_b : {self.argument_b}")

def main():
    rclpy.init()
    node = Calculator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()

if __name__ == "__main__":
    main()
