import time

import RPi.GPIO as gpio

SERVO_PIN = 13
gpio.setmode(gpio.BCM)
gpio.setup(SERVO_PIN, gpio.OUT)

pwm = gpio.PWM(SERVO_PIN, 50)
pwm.start(0)

def set_servo_angle(angle):
    duty_cycle = 2.5 + (angle/18.0)
    pwm.ChangeDutyCycle(duty_cycle)
    time.sleep(0.5)
    pwm.ChangeDutyCycle(0)
    
for i in range(11):
    set_servo_angle(i * 18)
    time.sleep(1)

gpio.cleanup()
