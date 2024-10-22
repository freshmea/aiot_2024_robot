import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool


class Service_server(Node):
    def __init__(self):
        super().__init__("hello_pub")
        self.create_service(SetBool, "setBool", self.setBool_callback)
        self.bool = bool()

    def setBool_callback(self, request : SetBool.Request, response : SetBool.Response):
        self.get_logger().info(f"{request.data}")
        self.get_logger().info(f"{self.bool}")
        if request.data != self.bool:
            self.bool = not self.bool
            response.success = True
            response.message = f"{self.bool} setting sucess"
        else:
            response.success = False
            response.message = f"{self.bool} setting fail"
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
