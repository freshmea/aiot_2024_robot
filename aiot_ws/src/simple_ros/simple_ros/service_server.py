import time

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool


class Service_server(Node):
    def __init__(self):
        super().__init__("service_server")
        self.create_service(SetBool, "setBool", self.setBool_callback)
        self.bool = bool()
        self.cnt = 0

    def setBool_callback(self, request: SetBool.Request, response: SetBool.Response):
        self.get_logger().info(f"{self.cnt}번째 요청 처리")
        self.get_logger().info(f"현재 세팅된 불 값 : {self.bool}")
        self.get_logger().info(f"변경 요청한 불 값 : {request.data}")
        if request.data != self.bool:
            self.bool = not self.bool
            response.success = True
            response.message = f"{self.cnt}번째 요청 {self.bool} setting success"
        else:
            response.success = False
            response.message = f"{self.cnt}번째 요청 {self.bool} setting fail"
        time.sleep(5)
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
