import time

import rclpy
import RPi.GPIO as gpio
from rclpy.node import Node
from std_msgs.msg import Float32


class ServoSub(Node):
    def __init__(self):
        super().__init__("servo_sub")
        self.create_subscription(Float32, "servo_angle", self.set_angle_callback, 10)
        self.servo_pin = 13
        gpio.setmode(gpio.BCM)
        gpio.setup(self.servo_pin, gpio.OUT)
        self.pwm = gpio.PWM(self.servo_pin, 50)
        self.pwm.start(0)
        self.theta = 0.0
        self.pwm.ChangeDutyCycle(2.5)
        time.sleep(1)
        self.pwm.ChangeDutyCycle(0)

    def set_angle_callback(self, msg: Float32):
        duty_cycle = 2.5 + (msg.data/3.14*180/18.0)
        self.pwm.ChangeDutyCycle(duty_cycle)
        self.theta = msg.data

def main():
    rclpy.init()
    node = ServoSub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        gpio.cleanup()
        node.destroy_node()

if __name__ == "__main__":
    main()
