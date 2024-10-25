import random

import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node, Parameter
from user_interface.msg import ArithmeticArgument


class Argument(Node):
    def __init__(self):
        super().__init__("argument")
        self.declare_parameter("min", 0)
        self.declare_parameter("max", 30)

        self.min : int = self.get_parameter("min").value #type: ignore
        self.max : int = self.get_parameter("max").value #type: ignore
        self.add_on_set_parameters_callback(self.update_parameter)
        self.create_timer(1.0, self.pub)
        self.pub_o = self.create_publisher(ArithmeticArgument, "arithmetic_argument", 10)

    def pub(self):
        msg = ArithmeticArgument()
        msg.stamp = self.get_clock().now().to_msg()
        msg.argument_a = float(random.randint(self.min, self.max))
        msg.argument_b = float(random.randint(self.min, self.max))
        self.get_logger().info(f"Argument_a : {msg.argument_a}")
        self.get_logger().info(f"Argument_b : {msg.argument_b}")
        self.pub_o.publish(msg)

    def update_parameter(self, params: list[Parameter]):
        for param in params:
            if param.name == 'min':
                self.min = param.value #type: ignore
            if param.name == 'max':
                self.max = param.value #type: ignore
        return SetParametersResult(successful=True)

def main(args = None):
    rclpy.init(args = args)
    node = Argument()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()

if __name__ == "__main__":
    main()
