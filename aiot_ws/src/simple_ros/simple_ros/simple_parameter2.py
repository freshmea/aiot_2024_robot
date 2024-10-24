import rclpy
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node


class Simple_parameter(Node):
    def __init__(self):
        super().__init__("simple_parameter")
        self.client = self.create_client(SetParameters, "/simple_parameter/set_parameters")
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("server is not available!!")
        self.create_timer(1, self.print_parameter)
        self.count = 0
        self.declare_parameter('number1', 0)

    def print_parameter(self):
        self.count += 1
        param = Parameter()
        param.name = 'number1'
        param.value = ParameterValue()
        param.value.type = ParameterType.PARAMETER_INTEGER
        param.value.integer_value = self.count

        req = SetParameters.Request()
        req.parameters = [param]
        future = self.client.call_async(req)
        future.add_done_callback(self.done_callback)
    
    def done_callback(self, future):
        response = future.result()
        self.get_logger().info(f"{response.results}")

def main():
    rclpy.init()
    node = Simple_parameter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()

if __name__ == "__main__":
    main()
