import rclpy
import serial
from rclpy.node import Node
from std_msgs.msg import String


class Arduino_lcd(Node):
    def __init__(self):
        super().__init__("arduino_led")
        self.create_subscription(String, "lcd", self.sub_callback, 10)
        self.ser = serial.Serial('/dev/ttyACM0', 115200)

    def sub_callback(self, msg: String):
        if len(msg.data) < 20:
            msg.data = msg.data + ' '*(20-len(msg.data))
        elif len(msg.data) > 20:
            msg.data = msg.data[:20]

        if msg.data[:3] == 'lcd':
            byte_msg = (msg.data+'\n').encode('utf-8')
            self.ser.write(byte_msg)
            self.get_logger().info(msg.data)

def main():
    rclpy.init()
    node = Arduino_lcd()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == "__main__":
    main()
