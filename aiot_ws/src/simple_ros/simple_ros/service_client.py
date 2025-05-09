import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool


class Service_client(Node):
    def __init__(self):
        super().__init__("service_client")
        self.client = self.create_client(SetBool, "setBool")
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("service not available")
        self.request = SetBool.Request()
        self.create_timer(1, self.update)
        self.create_timer(2, self.send_request)
        self.cnt = 0
        self.send_request()

    def send_request(self):
        self.cnt += 1
        self.get_logger().info(f"{self.cnt} 번째 요청")
        self.request.data = not self.request.data
        self.future = self.client.call_async(self.request)
        self.future.add_done_callback(self.done_callback)

    def done_callback(self, future):
        response: SetBool.Response = future.result()
        self.get_logger().info(f"{response.success}")
        self.get_logger().info(response.message)

    def update(self):
        self.get_logger().info("updating!!")


def main():
    rclpy.init()
    node = Service_client()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()


if __name__ == "__main__":
    main()
