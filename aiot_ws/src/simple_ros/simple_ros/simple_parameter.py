import rclpy
from rclpy.node import Node


class Simple_parameter(Node):
    def __init__(self):
        super().__init__("hello_pub")
        self.declare_parameter('my_para', '내가 만든 파라미터')
        self.my_para = self.get_parameter('my_para').get_parameter_value().string_value
        self.create_timer(1, self.print_parameter)

    def print_parameter(self):
        self.get_logger().info(f"parameter : {self.my_para}")

def main():
    rclpy.init()
    node = Simple_parameter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()

if __name__ == "__main__":
    main()
