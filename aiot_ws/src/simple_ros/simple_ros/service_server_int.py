import rclpy
from rclpy.node import Node
from user_interface.srv import AddAndOdd


class Service_server(Node):
    def __init__(self):
        super().__init__("service_server")
        self.create_service(AddAndOdd, "addandodd", self.service_callback)
        self.bool = bool()

    def service_callback(
        self, request: AddAndOdd.Request, response: AddAndOdd.Response
    ):
        response.stamp = self.get_clock().now().to_msg()
        response.sum = request.inta + request.intb
        if response.sum % 2:
            response.odd = "two ints sum is odd"
        else:
            response.odd = "two ints sum is not odd"
        return response


def main():
    rclpy.init()
    node = Service_server()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()


if __name__ == "__main__":
    main()
