import time

import rclpy
import RPi.GPIO as gpio
from rclpy.node import Node
from tb_interface.srv import ServoAngle


class ServoServer(Node):
    def __init__(self):
        super().__init__("servo_server")
        self.create_service(ServoAngle, "servo_angle", self.set_angle_callback)
        self.bool = bool()
        self.servo_pin = 13
        gpio.setmode(gpio.BCM)
        gpio.setup(self.servo_pin, gpio.OUT)
        self.pwm = gpio.PWM(self.servo_pin, 50)
        self.pwm.start(0)
        self.theta = 0.0
        self.pwm.ChangeDutyCycle(2.5)
        time.sleep(1)
        self.pwm.ChangeDutyCycle(0)

    def set_servo_angle(self, angle, seconds):
        duty_cycle = 2.5 + (angle/3.14*180/18.0)
        self.pwm.ChangeDutyCycle(duty_cycle)
        time.sleep(seconds)
        self.pwm.ChangeDutyCycle(0)
        self.theta = angle

    def set_angle_callback(self, request : ServoAngle.Request, response : ServoAngle.Response):
        ITER_NUM = 10
        for i in range(ITER_NUM):
            self.set_servo_angle(self.theta -(self.theta - request.theta)/ITER_NUM*(i+1), request.seconds/ITER_NUM)
        if self.theta == request.theta:
            response.delta_theta = self.theta - request.theta
            response.success = True
        else:
            response.delta_theta = self.theta - request.theta
            response.success = False
        return response

def main():
    rclpy.init()
    node = ServoServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        gpio.cleanup()
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == "__main__":
    main()
