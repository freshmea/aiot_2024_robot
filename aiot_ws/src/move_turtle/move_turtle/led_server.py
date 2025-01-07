import time

import rclpy
import RPi.GPIO as gpio
from rclpy.node import Node
from std_srvs.srv import SetBool


class LedServer(Node):
    def __init__(self):
        super().__init__("service_server")
        self.create_service(SetBool, "led_on_off", self.setBool_callback)
        self.bool = bool()
        gpio.setmode(gpio.BCM)
        gpio.setup(21, gpio.OUT)

    def setBool_callback(self, request : SetBool.Request, response : SetBool.Response):
        self.get_logger().info(f"request {request.data}")
        self.get_logger().info(f"internal bool {self.bool}")
        if request.data != self.bool:
            self.bool = not self.bool
            if self.bool:
                gpio.output(21, gpio.HIGH)
            else:
                gpio.output(21, gpio.LOW)
            response.success = True
            response.message = f"{self.bool} setting sucess"
        else:
            response.success = False
            response.message = f"{self.bool} setting fail"
        return response

def main():
    rclpy.init()
    node = LedServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        gpio.cleanup()
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == "__main__":
    main()
