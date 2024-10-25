import random

import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node, Parameter
from rclpy.task import Future
from user_interface.srv import ArithmeticOperator


class Operator(Node):
    def __init__(self):
        super().__init__("operator")
        self.declare_parameter("service_time", 5)
        self.service_time = self.get_parameter("service_time").get_parameter_value().integer_value
        self.add_on_set_parameters_callback(self.update_parameter)
        self.future = Future()
        self.client = self.create_client(ArithmeticOperator, "arithmetic_operator")
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("service is not available!")
        self.create_timer(self.service_time, self.send_request)

    def send_request(self):
        request = ArithmeticOperator.Request()
        request.arithmetic_operator = random.randint(1,4)
        self.future = self.client.call_async(request)
        self.future.add_done_callback(self.done_callback)

    def done_callback(self, future):
        result : ArithmeticOperator.Response = future.result()
        self.get_logger().info(f"{result.arithmetic_result}")

    def update_parameter(self, params: list[Parameter]):
        for param in params:
            if param.name == 'service_time':
                self.service_time = param.value #type: ignore
        return SetParametersResult(successful=True)

def main():
    rclpy.init()
    node = Operator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()

if __name__ == "__main__":
    main()
