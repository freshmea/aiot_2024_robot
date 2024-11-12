import time

import RPi.GPIO as gpio

gpio.setmode(gpio.BCM)
gpio.setup(21, gpio.OUT)

for _ in range(10):
    gpio.output(21, gpio.HIGH)
    time.sleep(1)
    gpio.output(21, gpio.LOW)
    time.sleep(1)
gpio.cleanup()
    