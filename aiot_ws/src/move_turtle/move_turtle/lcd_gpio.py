import time
from time import *

import rclpy
import RPi_I2C_driver
from rclpy.node import Node
from tb_interface.srv import LcdDisplay


class LcdServer(Node):
    def __init__(self):
        super().__init__("lcd_server")
        self.create_service(LcdDisplay, "lcd_server", self.set_display_callback)
        # lcd init
        self.lcd = RPi_I2C_driver.lcd(0x27)
        self.lcd.cursor()

    def set_display_callback(self, request : LcdDisplay.Request, response : LcdDisplay.Response):
        self.get_logger().info(f"request {request.data}")
        # lcd write
        str = f"{request.data}"
        if str[0:3] == "lcd":
            if str[3] == "0":
                lcd.setCursor(0,0)
            if str[3] == "1":
                lcd.setCursor(0,1)
            if str[3] == "2":
                lcd.setCursor(0,2)
            if str[3] == "3":
                lcd.setCursor(0,3)
            lcd.print(f"request {request.data[4:]}")
        
        response.success = True
        return response

def main():
    rclpy.init()
    node = LcdServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        gpio.cleanup()
        node.destroy_node()

if __name__ == "__main__":
    main()
