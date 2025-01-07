import rclpy
import serial
from rclpy.node import Node
from std_msgs.msg import String


class Arduino_switch(Node):
    def __init__(self):
        super().__init__("arduino_led")
        self.pub = self.create_publisher(String, "switch", 10)
        self.create_timer(1/60, self.pub_callback)
        self.ser = serial.Serial('/dev/ttyACM0', 115200)

    def pub_callback(self):
        msg = String()
        msg.data = self.ser.read_until(b'\n').decode('utf-8').replace('\r', '').replace('\n', '')
        self.ser.flush()
        self.get_logger().info(msg.data)
        self.pub.publish(msg)

def main():
    rclpy.init()
    node = Arduino_switch()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == "__main__":
    main()
