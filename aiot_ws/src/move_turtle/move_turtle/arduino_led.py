import rclpy
import serial
from rclpy.node import Node
from std_msgs.msg import String


class Arduino_led(Node):
    def __init__(self):
        super().__init__("arduino_led")
        self.create_subscription(String, "led", self.sub_callback, 10)
        self.ser = serial.Serial('/dev/ttyACM0', 115200)

    def sub_callback(self, msg: String):
        byte_msg = msg.data.encode('utf-8')
        self.ser.write(byte_msg)
        self.get_logger().info(msg.data)

def main():
    rclpy.init()
    node = Arduino_led()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()

if __name__ == "__main__":
    main()
