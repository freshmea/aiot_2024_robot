import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter


class Simple_parameter(Node):
    def __init__(self):
        super().__init__("hello_pub")
        self.declare_parameter('my_para', '내가 만든 파라미터')
        self.add_on_set_parameters_callback(self.parameter_callback)
        self.my_para = self.get_parameter('my_para').get_parameter_value().string_value
        self.create_timer(1, self.print_parameter)

    def print_parameter(self):
        self.get_logger().info(f"parameter : {self.my_para}")
    
    def parameter_callback(self, parameters : list[Parameter]):
        for parameter in parameters:
            if parameter.name == 'my_para':
                # self.my_para = parameter.get_parameter_value().string_value
                self.my_para = parameter.value
        return SetParametersResult(successful=True)

def main():
    rclpy.init()
    node = Simple_parameter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()

if __name__ == "__main__":
    main()
