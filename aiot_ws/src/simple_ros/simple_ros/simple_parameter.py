import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter


class Simple_parameter(Node):
    def __init__(self):
        super().__init__("simple_parameter")
        self.declare_parameter('my_para', '내가 만든 파라미터')
        self.declare_parameter('node_name', 'simple_parameter')
        self.declare_parameter('number1', 123)
        self.declare_parameter('number2', 3.141592)

        self.my_para = self.get_parameter('my_para').get_parameter_value().string_value
        self.node_name = self.get_parameter('node_name').get_parameter_value().string_value
        self.number1 = self.get_parameter('number1').get_parameter_value().integer_value
        self.number2 = self.get_parameter('number2').get_parameter_value().double_value

        self.add_on_set_parameters_callback(self.parameter_callback)

        self.create_timer(1, self.print_parameter)

    def print_parameter(self):
        self.get_logger().info(f"parameter : {self.my_para}")
        self.get_logger().info(f"parameter : {self.node_name}")
        self.get_logger().info(f"parameter : {self.number1}")
        self.get_logger().info(f"parameter : {self.number2}")

    def parameter_callback(self, parameters : list[Parameter]):
        for parameter in parameters:
            if parameter.name == 'my_para':
                # self.my_para = parameter.get_parameter_value().string_value
                self.my_para = parameter.value
            if parameter.name == 'node_name':
                self.node_name = parameter.value
            if parameter.name == 'number1':
                self.number1 = parameter.value
            if parameter.name == 'number2':
                self.number2 = parameter.value
        return SetParametersResult(successful=True)

def main():
    rclpy.init()
    node = Simple_parameter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == "__main__":
    main()
