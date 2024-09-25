from threading import Thread, Lock
from smbus import * 
from spidev import * 
from bme680 import *
import math
import time
import RPi.GPIO as GPIO
import spidev

#for Camera (AutoCar, Serbot)
import traitlets, os, cv2
from traitlets.config.configurable import SingletonConfigurable
import atexit
from IPython.display import display
import ipywidgets.widgets as widgets
import __main__

#for PixelDisplay
import board
from neopixel_spi import NeoPixel_SPI

__main__._camera_flip_method = '0'

import wave
import pyaudio
import audioop
import numpy as np

##################################################################
#binder = ctypes.cdll.LoadLibrary('/usr/local/lib/libpop.so')

##################################################################

GPIO.setmode(GPIO.BCM)

def checkI2C(bus, addr):
    try:
        bus = SMBus(bus)
        bus.write_byte(addr, 0)
    except Exception as e:
        if not hasattr(e, "errno") or e.errno != 121:
            print(e)
        return False
    return True

_cat = 0

if checkI2C(0,0x5d): #AutoCar
    __main__._camera_flip_method='0'
elif checkI2C(0,0x40): #SerBot
    __main__._camera_flip_method='2'
    _cat = 2
elif checkI2C(0,0x5c): #AutoCar Prime
    __main__._camera_flip_method='0'
    _cat = 3
elif checkI2C(1,0x5c): #AutoCar Prime + NX
    __main__._camera_flip_method='0'
    _cat = 6
else:
    try:
        import can
        __c=can.interface.Bus(channel='can0', bustype='socketcan_native')
        __c.send(can.Message(arbitration_id=0x100,is_extended_id=False,data=None))
        __splitter=__c.recv(timeout=2).data[0]

        if __splitter==1: #AutoCar Prime X
            _cat = 5
            __main__._camera_flip_method='0'
        elif __splitter==2: #SerBot Prime X
            _cat = 4
            __main__._camera_flip_method='2'
        else:
            _cat = 4 #Default : SerBot Prime X
            __main__._camera_flip_method='2'
        __c=None
    except Exception as e:
        try:
            if e.errno==19:
                _cat = 1 #AutoCar Racing
                __main__._camera_flip_method='0'
        except:
            pass


##################################################################

def map(x, in_min, in_max, out_min, out_max):
    return ((x - in_min) * (out_max - out_min) // (in_max - in_min) + out_min)
def delay(millisecond):
    time.sleep(millisecond / 1000)
##################################################################
class PopThread:
    def __init__(self):
        self._state = False    
        self._thread = None

    def __raw_run(self):
        try:
            while (self.isRun()):
                self.run()
                time.sleep(0.02) #ony Python thread
        finally:
            pass

    def start(self, daemon=True):
        self._state = True
        self._thread = Thread(target=self.__raw_run)
        self._thread.daemon = daemon
        self._thread.start()

    def run(self):
        pass

    def stop(self):
        self._state = False

    def isRun(self):
        return self._state
        
class My(PopThread):
    def __init__(self):
        self.i = 0

    def run(self):
        self.i += 1
        print("run: %d"%(self.i))
        time.sleep(0.02)

def popmultitask_unittest():
    my = My()
    my.start()
    for i in range(10):
        print("terminate: %d"%(i+1))
        time.sleep(0.02)
    my.stop()
    my.start()
    for i in range(9, -1, -1):
        print("terminate: %d"%(i+1))
        time.sleep(0.02)
    my.stop()
##################################################################

class Out:
    def __init__(self, n):
        self._gpio = n
        GPIO.setup(self._gpio, GPIO.OUT)

    def __del__(self):
        self.off()

    def on(self):
        GPIO.output(self._gpio, GPIO.HIGH)

    def off(self):
        GPIO.output(self._gpio, GPIO.LOW)

##################################################################
class Led(Out):
    def __init__(self, n):
        super().__init__(n)

    def __del__(self):
        super().__del__()

def led_unittest():
    leds = [Led(23), Led(24), Led(25), Led(1)]

    for _ in range(100):
        for i in range(4):
            leds[i].on()
            time.sleep(0.01)
        
        for i in range(4):
            leds[i].off()
            time.sleep(0.01)

##################################################################
class Laser(Out):
    def __init__(self, n):
        super().__init__(n)

    def __del(self):
        super().__del__()

def laser_unittest():
    laser = Laser(2)

    for i in range(10):
        laser.on()
        time.sleep(0.1)
        laser.off()
        time.sleep(0.1)
##################################################################

class Buzzer(Out):

    def __init__(self, n):
        super().__init__(n)

    def __del__(self):
        super().__del__()

def buzzer_unittest():
    buzzer = Buzzer(2)

    for i in range(10):
        buzzer.on()
        time.sleep(0.05)
        buzzer.off()
        time.sleep(0.05)
##################################################################

class Relay(Out):

    def __init__(self, n):
        super().__init__(n)

    def __del__(self):
        super().__del__()

def relay_unittest1():
    relay = Relay(2)

    for _ in range(10):
        relay.on()
        time.sleep(1)
        relay.off()
        time.sleep(1)

def relay_unittest2():
    relay = Relay(2)

    while (True):
        relay.on()
        input("Press <ENTER> key...\n")
        relay.off()
        input("Press <ENTER> key...\n")
##################################################################
# LedEx

##################################################################

class RGBLed:
    RED_COLOR = 1
    GREEN_COLOR = 2
    BLUE_COLOR = 4

    def __init__(self, *ns):
        self._gpios = ns
        for gpio in self._gpios:
            pinMode(gpio, OUTPUT)
            softPwmCreate(gpio, 0, 255)

    def __del__(self):
        for gpio in self._gpios:
            softPwmWrite(gpio, 0)

    def on(self, color):
        for i, gpio in enumerate(self._gpios):
            if ((color >> i) & 0x01):
                softPwmWrite(gpio, 255)

    def off(self, color):
        for i, gpio in enumerate(self._gpios):
            if ((color >> i) & 0x01):
                softPwmWrite(gpio, 0)

    def set(self, *colors):
        l = len(colors)

        if l == 1:
            for i, gpio in enumerate(self._gpios):
                if ((colors[0] >> i) & 0x01):
                    softPwmWrite(gpio, 255)
                else:
                    softPwmWrite(gpio, 0)
        elif l == 3:
            for i, color in enumerate(colors):
                softPwmWrite(self._gpios[i], color)


def rgbled_unittest1():
    rgbled = RGBLed(2, 3, 4)

    rgbled.on(RGBLed.RED_COLOR | RGBLed.GREEN_COLOR | RGBLed.BLUE_COLOR)
    time.sleep(1)
    rgbled.off(RGBLed.BLUE_COLOR)
    time.sleep(1)
    rgbled.off(RGBLed.GREEN_COLOR)
    time.sleep(1)
    rgbled.off(RGBLed.RED_COLOR)
    time.sleep(1)
    rgbled.set(RGBLed.RED_COLOR)
    time.sleep(1)
    rgbled.set(RGBLed.GREEN_COLOR)
    time.sleep(1)
    rgbled.set(RGBLed.BLUE_COLOR)
    time.sleep(1)

def rgbled_unittest2():
    rgbled = RGBLed(2, 3, 4)

    while (True):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        print("R=%03d, G=%03d, B=%03d"%(r, g, b))
        rgbled.set(r, g, b)
        time.sleep(0.1)


def rgbled_unittest3():
    rgbled = RGBLed(2, 3, 4)

    while (True):
        r = int(input("Enter Red(0 ~ 255): "))
        g = int(input("Enter Green(0 ~ 255): "))
        b = int(input("Enter Blue(0 ~ 255): "))
        print(">>> set color: %3d:%3d:%3d"%(r, g, b))
        rgbled.set(r, g, b)
        time.sleep(0.1)

##################################################################

class DCMotor(object):
    SPEED_1 = 3
    SPEED_2 = 5
    SPEED_3 = 10
    def __init__(self, *ns):
        self._gpios = ns
        self._speed = 0

        for gpio in self._gpios:
            pinMode(gpio, OUTPUT)
            softPwmCreate(gpio, 0, 10)

    def __del__(self):
        self.stop(False)

        for gpio in self._gpios:
            digitalWrite(gpio, LOW)

    def forward(self):
        softPwmWrite(self._gpios[0], self._speed)
        softPwmWrite(self._gpios[1], 0)

    def backward(self):
        softPwmWrite(self._gpios[0], 0)
        softPwmWrite(self._gpios[1], self._speed)

    def setSpeed(self, speed):
        self._speed = speed

    def stop(self, isBreak=True):
        for gpio in self._gpios:
            if (isBreak):
                softPwmWrite(gpio, 10)
            else:
                softPwmWrite(gpio, 0)

def dcmotor_unittest1():
    dcmotor = DCMotor(2, 3)
    dcmotor.setSpeed(DCMotor.SPEED_3)
    dcmotor.forward()
    time.sleep(3)
    dcmotor.backward()
    time.sleep(3)

def dcmotor_unittest2():
    dcmotor = DCMotor(2, 3)
    speeds = [DCMotor.SPEED_1, DCMotor.SPEED_2, DCMotor.SPEED_3]

    for speed in speeds:
        dcmotor.setSpeed(speed)
        dcmotor.forward()
        input("Press <ENTER> key...\n")

    dcmotor.stop()
    time.sleep(2)
##################################################################
# Step Motor 
##################################################################
# DC Fan
class Fan(Out):
    def __init__(self, n):
        super().__init__(n)

    def __del__(self):
        super().__del__()

def fan_unittest():
    fan = Fan(26)
    for i in range(10):
        if i%2 == 0:
            fan.on()
        else:
            fan.off()

##################################################################

class Input:
    FALLING = GPIO.FALLING
    RISING = GPIO.RISING
    BOTH = GPIO.BOTH

    def __init__(self, n, activeHigh=True):
        self._gpio = n
        self._func = None
        self._param = None
        self._activeHigh = activeHigh

        GPIO.setup(self._gpio, GPIO.IN)

    def read(self):
        level = GPIO.input(self._gpio)
        return level if self._activeHigh else not level

    def setCallback(self, func, param=None, type=BOTH):
        self._func = func
        self._param = param
        #GPIO.add_event_detect(self._gpio, type)
        #return GPIO.add_event_detect(self._gpio, type, callback=self._wrapper if self._func != None else self._dummy)
        return GPIO.add_event_detect(self._gpio, type, callback=self._func if self._func != None else self._dummy)

    def _wrapper(self):
        self._func(self._param)

    def _dummy(self):
        pass

##################################################################

class Pir(Input):
    def __init__(self, n):
        super().__init__(n)

##################################################################

class SpiAdc(PopThread):
    MODE_INCLUSIVE = 1
    MODE_EXCLUSIVE = 2
    MODE_FULL = 3

    TYPE_NORMAL = 1
    TYPE_AVERAGE = 2

    REF_VOLTAG = 3.3
    ADC_MAX = 4096 - 1

    def __init__(self, channel, device=0, bus=0, speed=500000):
        self._channel = channel
        self._device = device
        self._bus = bus
        self._spi = spidev.SpiDev()
        self._spi.open(0,0)
        self._spi.max_speed_hz = speed
        self._func = None
        self._param = None
        self._sample = 1
        self._type = None
        self._mode = None
        self._min = None
        self._max = None

        
    def setCallback(self, func, param=None, type=TYPE_AVERAGE, mode=MODE_FULL, min=0, max=ADC_MAX):
        self._func = func
        self._param = param
        self._type = type
        self._mode = mode
        self._min = min
        self._max = max

        self.start() if (self._func != None) else self.stop()

    def setSample(self, sample):
        self._sample = sample

    def getSample(self):
        return self._sample

    def read(self):
        r = self._spi.xfer2([6|(self._channel>>2), (self._channel & 3) << 6, 0])
        adcval = ((r[1] & 15) << 8) + r[2]
        return adcval

    def readAverage(self):
        val = 0.0

        for _ in range(self._sample):
            val += math.pow(self.read(), 2)

        return int(math.sqrt(val / self._sample))

    def readVolt(self, ref=3.3, max=4095.0):
        return ref * (SpiAdc.read(self) / max)

    def readVoltAverage(self, ref=3.3, max=4095.0):
        return ref * (SpiAdc.readAverage(self) / max)

    def run(self):
        val = self.read() if (self._type == SpiAdc.TYPE_NORMAL) else self.readAverage()

        if (self._mode == SpiAdc.MODE_INCLUSIVE):
            if (val >= self._min and val <= self._max):
                self._func(val, self._param)
        elif (self._mode == SpiAdc.MODE_EXCLUSIVE):
            if (val < self._min or val > self._max):
                self._func(val, self._param)
        else:
            self._func(val, self._param)

##################################################################

class Cds(SpiAdc):
    def __init__(self, channel, device=0, bus=0, speed=500000):
        super().__init__(channel, device, bus, speed)
        self._funcPseudoLx = None

        self.setSample(1024)

    def setCalibrationPseudoLx(self, func):
        self._funcPseudoLx = func

    def readAverage(self):
        return self._calcPseudoLx(self.readVoltAverage())

    def _calcPseudoLx(self, volt):
        r = (10 * 33) / volt - 10
        lx = (129.15 * math.pow(r, -10.0 / 9)) * 1000

        if (self._funcPseudoLx != None):
            val = self._funcPseudoLx(volt, r, lx)
        else:
            if (lx >= 100 and lx < 190):
                lx *= 2
            elif (lx >= 190):
                lx *= 2.4

            val = int(math.floor(lx + 0.5))

        return val

##################################################################

class Gas(SpiAdc):
    def __init__(self, channel, device=0, bus=0, speed=500000):
        super().__init__(channel, device, bus, speed)       
        self.setSample(64)

        self._propan = [2.48, -0.64, -0.71]
        self._methan = [2.48, -0.50, -0.56]
        self._ethanol = [2.48, -0.35, -0.49]

        self._r0 = self.calibration()

    def calibration(self, rl=4.7, clean=1):
        val = 0.0

        for _ in range(32):
            val += self.resistanceCalculation(self.read(), rl)
            time.sleep(0.05)

        val /= 32

        return val / clean

    def setPropanCurve(self, x, y, inclination):
        self._propan = [x, y, inclination]

    def setMethanCurve(self, x, y, inclination):
        self._methan = [x, y, inclination]

    def setEthanolCurve(self, x, y, inclination):
        self._ethanol = [x, y, inclination]

    def calcPropan(self, val):
        ppm = self.resistanceCalculation(val) / self._r0

        return math.pow(10, (((math.log(ppm) - self._propan[1]) / self._propan[2]) + self._propan[0]))

    def calcMethan(self, val):
        ppm = self.resistanceCalculation(val) / self._r0

        return math.pow(10, (((math.log(ppm) - self._methan[1]) / self._methan[2]) + self._methan[0]))

    def calcEthanol(self, val):
        ppm = self.resistanceCalculation(val) / self._r0

        return math.pow(10, (((math.log(ppm) - self._ethanol[1]) / self._ethanol[2]) + self._ethanol[0]))

    def resistanceCalculation(self, val, rl=4.7):
        return (rl * (SpiAdc.ADC_MAX - val)) / val

##################################################################

class I2c():
    def __init__(self,addr,bus=8):
        self._sAddr = addr 
        self._bus = SMBus(bus)

    def __del__(self):
        self._bus.close()
        self._sAddr = 0
        self._bus = 0

    def read(self):
        return self._bus.read_byte(self._sAddr)

    def readByte(self, reg):
        return self._bus.read_byte_data(self._sAddr,reg)

    def readWord(self, reg):
        return self._bus.read_word_data(self._sAddr,reg)

    def readBlock(self, reg, length):
        return self._bus.read_i2c_block_data(self._sAddr,reg,length)

    def write(self, data):
        return self._bus.write_byte(self._sAddr,data)

    def writeByte(self, reg, data):
        return self._bus.write_byte_data(self._sAddr,reg,data)

    def writeWord(self, reg, data):
        return self._bus.write_word_data(self._sAddr,reg,data)

    def writeBlock(self, reg, data):
        return self._bus.write_i2c_block_data(self._sAddr,reg,data)

##################################################################
# PWM Controller IC 
class PwmController(I2c):
    PCA9685_REG_MODE1 = 0x00
    PCA9685_REG_MODE2 = 0x01
    PCA9685_REG_PRESCALE = 0xFE
    PCA9685_REG_LED0_ON_L = 0x06
    PCA9685_REG_LED0_ON_H = 0x07
    PCA9685_REG_LED0_OFF_L = 0x08
    PCA9685_REG_LED0_OFF_H = 0x09
    PCA9685_REG_ALL_ON_L = 0xFA
    PCA9685_REG_ALL_ON_H = 0xFB
    PCA9685_REG_ALL_OFF_L = 0xFC
    PCA9685_REG_ALL_OFF_H = 0xFD

    RESTART = 1<<7
    AI = 1<<5
    SLEEP = 1<<4
    ALLCALL	= 1<<0
    OCH = 1<<3
    OUTDRV = 1<<2
    INVRT = 1<<4

    def __init__(self, addr=0x5e):
        super().__init__(addr)
        self._curChannel = -1

    def __del(self):
        super().__del__()

    def init(self):
        buf = self.AI | self.ALLCALL
        I2c.writeByte(self,self.PCA9685_REG_MODE1,buf)
        buf = self.OCH | self.OUTDRV
        I2c.writeByte(self,self.PCA9685_REG_MODE2,buf)
        time.sleep(0.05)
        recv = I2c.readByte(self,self.PCA9685_REG_MODE1)
        buf = recv & (~self.SLEEP)
        I2c.writeByte(self,self.PCA9685_REG_MODE1,buf)

    def setChannel(self, ch):
        self._curChannel = ch

    def setDuty(self, percent):
        step = int(round(percent * (4096.0 / 100.0)))
        if step < 0:
            on = 0
            off = 4096
        elif step > 4095:
            on = 4096
            off = 0
        else:
            on = 0
            off = step

        on_l = on&0xff
        on_h = on>>8
        off_l = off&0xff
        off_h = off>>8
        if self._curChannel >= 0:
            I2c.writeByte(self,self.PCA9685_REG_LED0_ON_L+4*self._curChannel,on_l)
            I2c.writeByte(self,self.PCA9685_REG_LED0_ON_H+4*self._curChannel,on_h)
            I2c.writeByte(self,self.PCA9685_REG_LED0_OFF_L+4*self._curChannel,off_l)
            I2c.writeByte(self,self.PCA9685_REG_LED0_OFF_H+4*self._curChannel,off_h)
        elif self._curChannel == -1:
            I2c.writeByte(self,self.PCA9685_REG_ALL_ON_L,on_l)
            I2c.writeByte(self,self.PCA9685_REG_ALL_ON_H,on_h)
            I2c.writeByte(self,self.PCA9685_REG_ALL_OFF_L,off_l)
            I2c.writeByte(self,self.PCA9685_REG_ALL_OFF_H,off_h)

    def setFreq(self, freq):
        prescale = int(round(25000000/(4096*freq))-1)
        if prescale < 3:
            prescale = 3;
        elif prescale > 255:
            prescale = 255

        recv = I2c.readByte(self,self.PCA9685_REG_MODE1)
        buf = (recv &(~self.SLEEP))|self.SLEEP;
        I2c.writeByte(self,self.PCA9685_REG_MODE1,buf)
        I2c.writeByte(self,self.PCA9685_REG_PRESCALE,prescale)
        I2c.writeByte(self,self.PCA9685_REG_MODE1,recv)
        time.sleep(0.05)
        buf = recv | self.RESTART
        I2c.writeByte(self,self.PCA9685_REG_MODE1,buf)

    def setInvertPulse(self):
        recv = I2c.readByte(self,self.PCA9685_REG_MODE2)
        buf = recv | self.INVRT
        I2c.writeByte(self,self.PCA9685_REG_MODE2,buf)
        time.sleep(0.05)

def pca9685_unittest():
    pca9685 = Pca9685(0x5e)
    pca9685.init()
    pca9685.setChannel(0)
    pca9685.setDuty(0)
    pca9685.setFerq(200)
    for i in range(100):
        pca9685.setDuty(i)
        time.sleep(0.1)
    pca9685.setDuty(0)
    pca9685.setChannel(1)
    for i in range(100):
        pca9685.setDuty(i)
        time.sleep(0.1)
    pca9685.setDuty(0)
    pca9685.setChannel(2)
    for i in range(100):
        pca9685.setDuty(i)
        time.sleep(0.1)
    pca9685.setDuty(0)
    pca9685.setChannel(-1)
    for i in range(100):
        pca9685.setDuty(i)
        time.sleep(0.1)
    pca9685.setDuty(0)

##################################################################
# Touch Sensor for AIoT Connected Home 
class Touch(I2c):
    MPR121_TOUCHSTATUS_L   = 0x00
    MPR121_TOUCHSTATUS_H   = 0x01
    MPR121_FILTDATA_0L     = 0x04
    MPR121_FILTDATA_0H     = 0x05
    MPR121_BASELINE_0      = 0x1E
    MPR121_MHDR            = 0x2B
    MPR121_NHDR            = 0x2C
    MPR121_NCLR            = 0x2D
    MPR121_FDLR            = 0x2E
    MPR121_MHDF            = 0x2F
    MPR121_NHDF            = 0x30
    MPR121_NCLF            = 0x31
    MPR121_FDLF            = 0x32
    MPR121_NHDT            = 0x33
    MPR121_NCLT            = 0x34
    MPR121_FDLT            = 0x35
    MPR121_TOUCHTH_0       = 0x41
    MPR121_RELEASETH_0     = 0x42
    MPR121_DEBOUNCE        = 0x5B
    MPR121_CONFIG1         = 0x5C
    MPR121_CONFIG2         = 0x5D
    MPR121_ECR             = 0x5E
    MPR121_SOFTRESET       = 0x80

    def __init__(self, addr=0x5a):
        super().__init__(addr)
        self._channels = [None]*12
        self.reset()

    def __del__(self):
        super().__del__()

    def reset(self):
        I2c.writeByte(self,self.MPR121_SOFTRESET,0x63)
        time.sleep(0.1)
        I2c.writeByte(self,self.MPR121_ECR,0x00)
        data = I2c.readByte(self,self.MPR121_CONFIG2)
        if data != 0x24:
            print("MPR121 config ERROR MPR121_CONFIG2\n")
        for i in range(12):
            I2c.writeByte(self,self.MPR121_TOUCHTH_0+2*i,12)
            I2c.writeByte(self,self.MPR121_RELEASETH_0+2*i,6)
        I2c.writeByte(self,self.MPR121_MHDR,0x01)
        I2c.writeByte(self,self.MPR121_NHDR, 0x01)
        I2c.writeByte(self,self.MPR121_NCLR, 0x0E)
        I2c.writeByte(self,self.MPR121_FDLR, 0x00)
        I2c.writeByte(self,self.MPR121_MHDF, 0x01)
        I2c.writeByte(self,self.MPR121_NHDF, 0x05)
        I2c.writeByte(self,self.MPR121_NCLF, 0x01)
        I2c.writeByte(self,self.MPR121_FDLF, 0x00)
        I2c.writeByte(self,self.MPR121_NHDT, 0x00)
        I2c.writeByte(self,self.MPR121_NCLT, 0x00)
        I2c.writeByte(self,self.MPR121_FDLT, 0x00)
        I2c.writeByte(self,self.MPR121_DEBOUNCE, 0)
        I2c.writeByte(self,self.MPR121_CONFIG1, 0x10) 
        I2c.writeByte(self,self.MPR121_CONFIG2, 0x20) 
        I2c.writeByte(self,self.MPR121_ECR, 0x8F) 

    def readChannel(self,ch):
        value = 0 
        if ch >= 0 and ch < 8:
            value = I2c.readByte(self,0x00)
            value = (value >> ch) & 0x1 
        elif ch >= 8 and ch <12:
            value = I2c.readByte(self,0x01)
            value = (value >> (ch-8)) & 0x1
        self._channels[ch] = value
        return self._channels[ch]

    def readAll(self):
        value = I2c.readByte(self,0x00)
        for i in range(8):
            self._channels[i] = value & 0x01
            value = value >> 1
        value = I2c.readByte(self,0x01)
        for i in range(4):
            self._channels[i+8] = value & 0x01
            value = value >> 1
        return self._channels

##################################################################
# I2C Dust Sensor for AIoT Connected Home 
class Dust(I2c):
    def __init__(self, addr=0x28):
        super().__init__(addr)
        self._rBuf = [None]*32
        self.sensor_status = 0
        self.measuring_mode = 0
        self.calibration_factor = 0
        self.pm_1p0_grimm = 0
        self.pm_2p5_grimm = 0
        self.pm_10_grimm = 0
        self.pm_1p0_tsi = 0
        self.pm_2p5_tsi = 0
        self.pm_10_tsi = 0
        self.num_0p3 = 0
        self.num_0p5 = 0
        self.num_1 = 0
        self.num_2p5 = 0
        self.num_5 = 0
        self.num_10 = 0
        self.reset()

    def __del__(self):
        super().__del__()

    def reset(self):
        data = [0x16,0x7,0x03,0xff,0xff,0x00,0x16]
        for i in range(5):
            data[6] = data[i+1]
        I2c.writeBlock(self,0x50,data)

    def read(self):
        self._rBuf = I2c.readBlock(self,0x51,32)
        if self._rBuf[0] != 0x16:
            print("PM2008M Frame Heater is wrong!\n")
        else:
            if self._rBuf[1] != 32:
                print("PM2008M Frame Length is wrong!\n")
            else:
                check = self._rBuf[0]
                for i in range(30):
                    check = check ^ self._rBuf[i+1]
                if check != self._rBuf[31]:
                    print("PM2008M Check Code is wrong!\n")
                else:
                    self.sensor_status = self._rBuf[2]
                    self.measuring_mode = (self._rBuf[3]<<8)|self._rBuf[4]
                    self.calibration_factor = (self._rBuf[5]<<8)|self._rBuf[6]
                    self.pm_1p0_grimm = (self._rBuf[7]<<8)|self._rBuf[8]
                    self.pm_2p5_grimm = (self._rBuf[9]<<8)|self._rBuf[10]
                    self.pm_10_grimm = (self._rBuf[11]<<8)|self._rBuf[12]
                    self.pm_1p0_tsi = (self._rBuf[13]<<8)|self._rBuf[14]
                    self.pm_2p5_tsi = (self._rBuf[15]<<8)|self._rBuf[16]
                    self.pm_10_tsi = (self._rBuf[17]<<8)|self._rBuf[18]
                    self.num_0p3 = (self._rBuf[19]<<8)|self._rBuf[20]
                    self.num_0p5 = (self._rBuf[21]<<8)|self._rBuf[22]
                    self.num_1 = (self._rBuf[23]<<8)|self._rBuf[24]
                    self.num_2p5 = (self._rBuf[25]<<8)|self._rBuf[26]
                    self.num_5 = (self._rBuf[27]<<8)|self._rBuf[28]
                    self.num_10 = (self._rBuf[29]<<8)|self._rBuf[30]

def pm2008m_unittest():
    pm2008m = Pm2008m()
    while True:
        data = pm2008m.read()
        print("\n")
        print("PM 1.0 GRIM  : %u ㎍/㎥"%pm2008m.pm_1p0_grimm)	 
        print("PM 2.5 GRIM  : %u ㎍/㎥"%pm2008m.pm_2p5_grimm)
        print("PM 10  GRIM  : %u ㎍/㎥"%pm2008m.pm_10_grimm)
        print("PM 1.0 TSI   : %u ㎍/㎥"%pm2008m.pm_1p0_tsi)
        print("PM 2.5 TSI   : %u ㎍/㎥"%pm2008m.pm_2p5_tsi)
        print("PM 10  TSI   : %u ㎍/㎥"%pm2008m.pm_10_tsi)
        print("Number of 0.3 ㎛ : %u pcs/0.1L"%pm2008m.num_0p3)
        print("Number of 0.5 ㎛ : %u pcs/0.1L"%pm2008m.num_0p5)
        print("Number of 1 ㎛ : %u pcs/0.1L"%pm2008m.num_1)
        print("Number of 2.5 ㎛ : %u pcs/0.1L"%pm2008m.num_2p5)
        print("Number of 5 ㎛ : %u pcs/0.1L"%pm2008m.num_5)
        print("Number of 10 ㎛ : %u pcs/0.1L"%pm2008m.num_10)
        time.sleep(1)

##################################################################

class TempHumi(BME680):
    # Heater control settings
    ENABLE_HEATER = 0x00
    DISABLE_HEATER = 0x08

    # Gas measurement settings
    DISABLE_GAS_MEAS = 0x00
    ENABLE_GAS_MEAS = 0x01
    
    # Over-sampling settings
    OS_NONE = 0
    OS_1X = 1
    OS_2X = 2
    OS_4X = 3
    OS_8X = 4
    OS_16X = 5

    # IIR filter settings
    FILTER_SIZE_0 = 0
    FILTER_SIZE_1 = 1
    FILTER_SIZE_3 = 2
    FILTER_SIZE_7 = 3
    FILTER_SIZE_15 = 4
    FILTER_SIZE_31 = 5
    FILTER_SIZE_63 = 6
    FILTER_SIZE_127 = 7

    # Power mode settings
    SLEEP_MODE = 0
    FORCED_MODE = 1

    # Run gas enable and disable settings
    RUN_GAS_DISABLE = 0
    RUN_GAS_ENABLE = 1
    
    # Settings selector
    OST_SEL = 1
    OSP_SEL = 2
    OSH_SEL = 4
    GAS_MEAS_SEL = 8
    FILTER_SEL = 16
    HCNTRL_SEL = 32
    RUN_GAS_SEL = 64
    NBCONV_SEL = 128
    GAS_SENSOR_SEL = GAS_MEAS_SEL | RUN_GAS_SEL | NBCONV_SEL

    # Number of conversion settings
    NBCONV_MIN = 0
    NBCONV_MAX = 9 # Was 10, but there are only 10 settings: 0 1 2 ... 8 9

    def __init__(self, addr=0x77, debug=False):
        BME680.__init__(self, addr,i2c_device=SMBus(8))
        
        #initialize for gas
        self.setGasHeaterTemperature(320)
        self.setGasHeaterDuration(150)
        self.selectGasHeaterProfile(0)
        
        self.temperature = 0
        self.pressure = 0
        self.humidity = 0
        self.gas_resistance = 0

        self._last_reading = 0
        self._min_refresh_time = 0.1

    def init(self):
        self.softReset()
        self.setPowerMode(self.SLEEP_MODE)

        self._getCalibrationData()
        
        self.setHumidityOversample(self.OS_2X)
        self.setPressureOversample(self.OS_4X)
        self.setTemperatureOversample(self.OS_8X)
        self.setFilter(self.FILTER_SIZE_3)
        self.setGasStatus(self.ENABLE_GAS_MEAS)
        self.setTempOffset(0)

    def _getCalibrationData(self):
        self._get_calibration_data()

    def softReset(self):
        self.soft_reset()

    def setTempOffset(self, value):
        self.set_temp_offset(value)

    def setHumidityOversample(self, value):
        self.set_humidity_oversample(value)

    def getHumidityOversample(self):
        return self.get_humidity_oversample()

    def setPressureOversample(self, value):
        self.set_pressure_oversample(value)

    def getPressureOversample(self):
        return self.get_pressure_oversample()

    def setTemperatureOversample(self, value):
        self.set_temperature_oversample(value)

    def getTemperatureOversample(self):
        return self.get_temperature_oversample()

    def setFilter(self, value):
        self.set_filter(value)

    def getFilter(self):
        return self.get_filter()

    def selectGasHeaterProfile(self, value):
        self.gas_heater_index = value
        self.select_gas_heater_profile(value)

    def getGasHeaterProfile(self):
        return self.get_gas_heater_profile()

    def setGasStatus(self, value):
        self.set_gas_status(value)

    def getGasStatus(self):
        return self.get_gas_status()

    def setGasHeaterProfile(self, temperature, duration, nb_profile=0):
        self.set_gas_heater_profile(temperature, duration, nb_profile)

    def setGasHeaterTemperature(self, value, nb_profile=0):
        self.gas_heater_temperature = value
        self.set_gas_heater_temperature(value, nb_profile)

    def setGasHeaterDuration(self, value, nb_profile=0):
        self.gas_heater_duration = value
        self.set_gas_heater_duration(value, nb_profile)

    def setPowerMode(self, value, blocking=True):
        self.set_power_mode(value, blocking)

    def getPowerMode(self):
        return self.get_power_mode()

    def getSensorData(self):
        if self.get_sensor_data():
            d = self.data
            
            self.temperature = d.temperature
            self.pressure = d.pressure
            self.humidity = d.humidity
            self.gas_resistance = d.gas_resistance

            self._last_reading = time.monotonic()

            return((self.temperature, self.pressure, self.humidity, self.gas_resistance))
        else:
            return [None] * 4
            

    def isTime(self):
        if time.monotonic() - self._last_reading < self._min_refresh_time:
            return False
        else:
            return True

    def getTemperature(self):
        if self.isTime():
            self.getSensorData()
        return self.temperature

    def getPressure(self):
        if self.isTime():
            self.getSensorData()
        return self.pressure
        
    def getHumidity(self):
        if self.isTime():
            self.getSensorData()
        return self.humidity
        
    def getGas(self):
        if self.isTime():
            self.getSensorData()
        return self.gas_resistance

##################################################################

class Textlcd(I2c):
    LCD_WIDTH = 16		
    LCD_CMD = 0x00
    LCD_CHR = 0x01
    LCD_LINE1 = 0x00
    LCD_LINE2 = 0x40
    LCD_CLEAR = 0x01
    LCD_HOME = 0x02
    LCD_DISPLAY = 0x04
    LCD_CURSOR = 0x02
    LCD_BLINKING = 0x01
    LCD_DISPLAY_SHIFT_R = 0x1C
    LCD_DISPLAY_SHIFT_L = 0x18
    LCD_CURSOR_SHIFT_R = 0x14
    LCD_CURSOR_SHIFT_L = 0x10
    LCD_ENTRY_MODE_SET = 0x06
    LCD_BACKLIGHT = 0x08
    ENABLE = 0x04
    E_PULSE = 500
    E_DELAY = 500
    def __init__(self, addr=0x27):
        super().__init__(addr)
        self.command(0x33)
        self.command(0x32)

        self.command(0x28)
        self.command(0x0F)
        self.command(0x06)
        self.command(0x01)
        time.sleep(0.1)

        self.display_status = 0x0F

        self.returnHome()
        #self.print("Text LCD Init")

    def __del__(self):
        self.display_status = 0x00
        #self.clear()

    def _byte(self, byte, mode):
        high_bit = mode | (byte & 0xF0) | self.LCD_BACKLIGHT
        low_bit = mode | ((byte << 4) & 0xF0) | self.LCD_BACKLIGHT
        self._enable(high_bit)
        self._enable(low_bit)

    def _enable(self, byte):
        time.sleep(0.005)
        I2c.write(self, byte | self.ENABLE)
        time.sleep(0.005)
        I2c.write(self, byte & ~self.ENABLE)
        time.sleep(0.005)
        
    def command(self, command):
        self._byte(command,self.LCD_CMD)

    def clear(self):
        self.command(self.LCD_CLEAR)

    def returnHome(self):
        self.command(self.LCD_HOME)

    def displayOn(self):
        self.display_status = self.display_status | self.LCD_DISPLAY
        self.command(self.display_status)

    def displayOff(self):
        self.display_status = self.display_status & ~self.LCD_DISPLAY
        self.command(self.display_status)

    def displayShiftR(self):
        self.command(self.LCD_DISPLAY_SHIFT_R)

    def displayShiftL(self):
        self.command(self.LCD_DISPLAY_SHIFT_L)

    def cursorOn(self, blinking):
        self.display_status = self.display_status | self.LCD_CURSOR
        if blinking == 1:
            self.display_status = self.display_status | self.LCD_BLINKING
        else:
            self.display_status = self.display_status & ~self.LCD_BLINKING

        self.command(self.display_status)

    def cursorOff(self):
        self.display_status = self.display_status & ~self.LCD_CURSOR
        self.display_status = self.display_status & ~self.LCD_BLINKING
        self.command(self.display_status)

    def cursorShiftR(self):
        self.command(self.LCD_CURSOR_SHIFT_R)

    def cursorShiftL(self):
        self.command(self.LCD_CURSOR_SHIFT_L)

    def entryModeSet(self):
        self.command(self.LCD_ENTRY_MODE_SET)

    def setCursor(self, x, y):
        if x > 15:
            x = 15
        if y >= 1:
            y = self.LCD_LINE2
        else:
            y = self.LCD_LINE1
        self.command(0x80 | (x+y))

    def data(self, data):
        self._byte(data, self.LCD_CHR)

    def print(self, str):
        for i in str:
            self.data(ord(i))


##############################
#Oled Oled
SSD1306_I2C_ADDRESS = 0x3C 
SH1106_I2C_ADDRESS = 0x3C 
OLED_SSD1306_I2C_128x32 = 0
OLED_SH1106_I2C_128x64  = 1

BLACK = 0
WHITE = 1

SSD_Command_Mode = 0x00  
SSD_Data_Mode = 0x40  
SSD_Set_Segment_Remap = 0xA0
SSD_Inverse_Display = 0xA7
SSD_Set_Muliplex_Ratio = 0xA8
SSD_Display_Off = 0xAE
SSD_Display_On = 0xAF
SSD_Set_ContrastLevel = 0x81
SSD_External_Vcc = 0x01
SSD_Internal_Vcc = 0x02
SSD_Set_Column_Address = 0x21
SSD_Set_Page_Address = 0x22
SSD_Activate_Scroll = 0x2F
SSD_Deactivate_Scroll = 0x2E
SSD_Right_Horizontal_Scroll = 0x26
SSD_Left_Horizontal_Scroll = 0x27

Scroll_Left = 0x00
Scroll_Right = 0x01
Scroll_2Frames = 0x07
Scroll_3Frames = 0x04
Scroll_4Frames = 0x05
Scroll_5Frames = 0x00
Scroll_25Frames = 0x06
Scroll_64Frames = 0x01
Scroll_128Frames = 0x02
Scroll_256Frames = 0x03

SSD1306_Entire_Display_Resume = 0xA4
SSD1306_Entire_Display_On = 0xA5
SSD1306_Normal_Display = 0xA6
SSD1306_Set_Display_Offset = 0xD3
SSD1306_Set_Com_Pins = 0xDA
SSD1306_Set_Vcomh_Deselect_Level = 0xDB
SSD1306_Set_Display_Clock_Div = 0xD5
SSD1306_Set_Precharge_Period = 0xD9
SSD1306_Set_Lower_Column_Start_Address = 0x00
SSD1306_Set_Higher_Column_Start_Address = 0x10
SSD1306_Set_Start_Line = 0x40
SSD1306_Set_Memory_Mode = 0x20
SSD1306_Set_Com_Output_Scan_Direction_Normal = 0xC0
SSD1306_Set_Com_Output_Scan_Direction_Remap = 0xC8
SSD1306_Charge_Pump_Setting = 0x8D
SSD1306_SET_VERTICAL_SCROLL_AREA = 0xA3
SSD1306_VERTICAL_AND_RIGHT_HORIZONTAL_SCROLL = 0x29
SSD1306_VERTICAL_AND_LEFT_HORIZONTAL_SCROLL = 0x2A

SH1106_Set_Page_Address = 0xB0

OLED_FONT = [
    0x00, 0x00, 0x00, 0x00, 0x00,
    0x3E, 0x5B, 0x4F, 0x5B, 0x3E,
    0x3E, 0x6B, 0x4F, 0x6B, 0x3E,
    0x1C, 0x3E, 0x7C, 0x3E, 0x1C,
    0x18, 0x3C, 0x7E, 0x3C, 0x18, 
    0x1C, 0x57, 0x7D, 0x57, 0x1C, 
    0x1C, 0x5E, 0x7F, 0x5E, 0x1C, 
    0x00, 0x18, 0x3C, 0x18, 0x00, 
    0xFF, 0xE7, 0xC3, 0xE7, 0xFF, 
    0x00, 0x18, 0x24, 0x18, 0x00, 
    0xFF, 0xE7, 0xDB, 0xE7, 0xFF, 
    0x30, 0x48, 0x3A, 0x06, 0x0E, 
    0x26, 0x29, 0x79, 0x29, 0x26, 
    0x40, 0x7F, 0x05, 0x05, 0x07, 
    0x40, 0x7F, 0x05, 0x25, 0x3F, 
    0x5A, 0x3C, 0xE7, 0x3C, 0x5A, 
    0x7F, 0x3E, 0x1C, 0x1C, 0x08, 
    0x08, 0x1C, 0x1C, 0x3E, 0x7F, 
    0x14, 0x22, 0x7F, 0x22, 0x14, 
    0x5F, 0x5F, 0x00, 0x5F, 0x5F, 
    0x06, 0x09, 0x7F, 0x01, 0x7F, 
    0x00, 0x66, 0x89, 0x95, 0x6A, 
    0x60, 0x60, 0x60, 0x60, 0x60, 
    0x94, 0xA2, 0xFF, 0xA2, 0x94, 
    0x08, 0x04, 0x7E, 0x04, 0x08, 
    0x10, 0x20, 0x7E, 0x20, 0x10, 
    0x08, 0x08, 0x2A, 0x1C, 0x08, 
    0x08, 0x1C, 0x2A, 0x08, 0x08, 
    0x1E, 0x10, 0x10, 0x10, 0x10, 
    0x0C, 0x1E, 0x0C, 0x1E, 0x0C, 
    0x30, 0x38, 0x3E, 0x38, 0x30, 
    0x06, 0x0E, 0x3E, 0x0E, 0x06, 
    0x00, 0x00, 0x00, 0x00, 0x00, 
    0x00, 0x00, 0x5F, 0x00, 0x00, 
    0x00, 0x07, 0x00, 0x07, 0x00, 
    0x14, 0x7F, 0x14, 0x7F, 0x14, 
    0x24, 0x2A, 0x7F, 0x2A, 0x12, 
    0x23, 0x13, 0x08, 0x64, 0x62, 
    0x36, 0x49, 0x56, 0x20, 0x50, 
    0x00, 0x08, 0x07, 0x03, 0x00, 
    0x00, 0x1C, 0x22, 0x41, 0x00, 
    0x00, 0x41, 0x22, 0x1C, 0x00, 
    0x2A, 0x1C, 0x7F, 0x1C, 0x2A, 
    0x08, 0x08, 0x3E, 0x08, 0x08, 
    0x00, 0x80, 0x70, 0x30, 0x00, 
    0x08, 0x08, 0x08, 0x08, 0x08, 
    0x00, 0x00, 0x60, 0x60, 0x00, 
    0x20, 0x10, 0x08, 0x04, 0x02, 
    0x3E, 0x51, 0x49, 0x45, 0x3E, 
    0x00, 0x42, 0x7F, 0x40, 0x00, 
    0x72, 0x49, 0x49, 0x49, 0x46, 
    0x21, 0x41, 0x49, 0x4D, 0x33, 
    0x18, 0x14, 0x12, 0x7F, 0x10, 
    0x27, 0x45, 0x45, 0x45, 0x39, 
    0x3C, 0x4A, 0x49, 0x49, 0x31, 
    0x41, 0x21, 0x11, 0x09, 0x07, 
    0x36, 0x49, 0x49, 0x49, 0x36, 
    0x46, 0x49, 0x49, 0x29, 0x1E, 
    0x00, 0x00, 0x14, 0x00, 0x00, 
    0x00, 0x40, 0x34, 0x00, 0x00, 
    0x00, 0x08, 0x14, 0x22, 0x41, 
    0x14, 0x14, 0x14, 0x14, 0x14, 
    0x00, 0x41, 0x22, 0x14, 0x08, 
    0x02, 0x01, 0x59, 0x09, 0x06, 
    0x3E, 0x41, 0x5D, 0x59, 0x4E, 
    0x7C, 0x12, 0x11, 0x12, 0x7C, 
    0x7F, 0x49, 0x49, 0x49, 0x36, 
    0x3E, 0x41, 0x41, 0x41, 0x22, 
    0x7F, 0x41, 0x41, 0x41, 0x3E, 
    0x7F, 0x49, 0x49, 0x49, 0x41, 
    0x7F, 0x09, 0x09, 0x09, 0x01, 
    0x3E, 0x41, 0x41, 0x51, 0x73, 
    0x7F, 0x08, 0x08, 0x08, 0x7F, 
    0x00, 0x41, 0x7F, 0x41, 0x00, 
    0x20, 0x40, 0x41, 0x3F, 0x01, 
    0x7F, 0x08, 0x14, 0x22, 0x41, 
    0x7F, 0x40, 0x40, 0x40, 0x40, 
    0x7F, 0x02, 0x1C, 0x02, 0x7F, 
    0x7F, 0x04, 0x08, 0x10, 0x7F, 
    0x3E, 0x41, 0x41, 0x41, 0x3E, 
    0x7F, 0x09, 0x09, 0x09, 0x06, 
    0x3E, 0x41, 0x51, 0x21, 0x5E, 
    0x7F, 0x09, 0x19, 0x29, 0x46, 
    0x26, 0x49, 0x49, 0x49, 0x32, 
    0x03, 0x01, 0x7F, 0x01, 0x03, 
    0x3F, 0x40, 0x40, 0x40, 0x3F, 
    0x1F, 0x20, 0x40, 0x20, 0x1F, 
    0x3F, 0x40, 0x38, 0x40, 0x3F, 
    0x63, 0x14, 0x08, 0x14, 0x63, 
    0x03, 0x04, 0x78, 0x04, 0x03, 
    0x61, 0x59, 0x49, 0x4D, 0x43, 
    0x00, 0x7F, 0x41, 0x41, 0x41, 
    0x02, 0x04, 0x08, 0x10, 0x20, 
    0x00, 0x41, 0x41, 0x41, 0x7F, 
    0x04, 0x02, 0x01, 0x02, 0x04, 
    0x40, 0x40, 0x40, 0x40, 0x40, 
    0x00, 0x03, 0x07, 0x08, 0x00, 
    0x20, 0x54, 0x54, 0x78, 0x40, 
    0x7F, 0x28, 0x44, 0x44, 0x38, 
    0x38, 0x44, 0x44, 0x44, 0x28, 
    0x38, 0x44, 0x44, 0x28, 0x7F, 
    0x38, 0x54, 0x54, 0x54, 0x18, 
    0x00, 0x08, 0x7E, 0x09, 0x02, 
    0x18, 0xA4, 0xA4, 0x9C, 0x78, 
    0x7F, 0x08, 0x04, 0x04, 0x78, 
    0x00, 0x44, 0x7D, 0x40, 0x00, 
    0x20, 0x40, 0x40, 0x3D, 0x00, 
    0x7F, 0x10, 0x28, 0x44, 0x00, 
    0x00, 0x41, 0x7F, 0x40, 0x00, 
    0x7C, 0x04, 0x78, 0x04, 0x78, 
    0x7C, 0x08, 0x04, 0x04, 0x78, 
    0x38, 0x44, 0x44, 0x44, 0x38, 
    0xFC, 0x18, 0x24, 0x24, 0x18, 
    0x18, 0x24, 0x24, 0x18, 0xFC, 
    0x7C, 0x08, 0x04, 0x04, 0x08, 
    0x48, 0x54, 0x54, 0x54, 0x24, 
    0x04, 0x04, 0x3F, 0x44, 0x24, 
    0x3C, 0x40, 0x40, 0x20, 0x7C, 
    0x1C, 0x20, 0x40, 0x20, 0x1C, 
    0x3C, 0x40, 0x30, 0x40, 0x3C, 
    0x44, 0x28, 0x10, 0x28, 0x44, 
    0x4C, 0x90, 0x90, 0x90, 0x7C, 
    0x44, 0x64, 0x54, 0x4C, 0x44, 
    0x00, 0x08, 0x36, 0x41, 0x00, 
    0x00, 0x00, 0x77, 0x00, 0x00, 
    0x00, 0x41, 0x36, 0x08, 0x00, 
    0x02, 0x01, 0x02, 0x04, 0x02, 
    0x3C, 0x26, 0x23, 0x26, 0x3C, 
    0x1E, 0xA1, 0xA1, 0x61, 0x12, 
    0x3A, 0x40, 0x40, 0x20, 0x7A, 
    0x38, 0x54, 0x54, 0x55, 0x59, 
    0x21, 0x55, 0x55, 0x79, 0x41, 
    0x21, 0x54, 0x54, 0x78, 0x41, 
    0x21, 0x55, 0x54, 0x78, 0x40, 
    0x20, 0x54, 0x55, 0x79, 0x40, 
    0x0C, 0x1E, 0x52, 0x72, 0x12, 
    0x39, 0x55, 0x55, 0x55, 0x59, 
    0x39, 0x54, 0x54, 0x54, 0x59, 
    0x39, 0x55, 0x54, 0x54, 0x58, 
    0x00, 0x00, 0x45, 0x7C, 0x41, 
    0x00, 0x02, 0x45, 0x7D, 0x42, 
    0x00, 0x01, 0x45, 0x7C, 0x40, 
    0xF0, 0x29, 0x24, 0x29, 0xF0, 
    0xF0, 0x28, 0x25, 0x28, 0xF0, 
    0x7C, 0x54, 0x55, 0x45, 0x00, 
    0x20, 0x54, 0x54, 0x7C, 0x54, 
    0x7C, 0x0A, 0x09, 0x7F, 0x49, 
    0x32, 0x49, 0x49, 0x49, 0x32, 
    0x32, 0x48, 0x48, 0x48, 0x32, 
    0x32, 0x4A, 0x48, 0x48, 0x30, 
    0x3A, 0x41, 0x41, 0x21, 0x7A, 
    0x3A, 0x42, 0x40, 0x20, 0x78, 
    0x00, 0x9D, 0xA0, 0xA0, 0x7D, 
    0x39, 0x44, 0x44, 0x44, 0x39, 
    0x3D, 0x40, 0x40, 0x40, 0x3D, 
    0x3C, 0x24, 0xFF, 0x24, 0x24, 
    0x48, 0x7E, 0x49, 0x43, 0x66, 
    0x2B, 0x2F, 0xFC, 0x2F, 0x2B, 
    0xFF, 0x09, 0x29, 0xF6, 0x20, 
    0xC0, 0x88, 0x7E, 0x09, 0x03, 
    0x20, 0x54, 0x54, 0x79, 0x41, 
    0x00, 0x00, 0x44, 0x7D, 0x41, 
    0x30, 0x48, 0x48, 0x4A, 0x32, 
    0x38, 0x40, 0x40, 0x22, 0x7A, 
    0x00, 0x7A, 0x0A, 0x0A, 0x72, 
    0x7D, 0x0D, 0x19, 0x31, 0x7D, 
    0x26, 0x29, 0x29, 0x2F, 0x28, 
    0x26, 0x29, 0x29, 0x29, 0x26, 
    0x30, 0x48, 0x4D, 0x40, 0x20, 
    0x38, 0x08, 0x08, 0x08, 0x08, 
    0x08, 0x08, 0x08, 0x08, 0x38, 
    0x2F, 0x10, 0xC8, 0xAC, 0xBA, 
    0x2F, 0x10, 0x28, 0x34, 0xFA, 
    0x00, 0x00, 0x7B, 0x00, 0x00, 
    0x08, 0x14, 0x2A, 0x14, 0x22, 
    0x22, 0x14, 0x2A, 0x14, 0x08, 
    0xAA, 0x00, 0x55, 0x00, 0xAA, 
    0xAA, 0x55, 0xAA, 0x55, 0xAA, 
    0x00, 0x00, 0x00, 0xFF, 0x00, 
    0x10, 0x10, 0x10, 0xFF, 0x00, 
    0x14, 0x14, 0x14, 0xFF, 0x00, 
    0x10, 0x10, 0xFF, 0x00, 0xFF, 
    0x10, 0x10, 0xF0, 0x10, 0xF0, 
    0x14, 0x14, 0x14, 0xFC, 0x00, 
    0x14, 0x14, 0xF7, 0x00, 0xFF, 
    0x00, 0x00, 0xFF, 0x00, 0xFF, 
    0x14, 0x14, 0xF4, 0x04, 0xFC, 
    0x14, 0x14, 0x17, 0x10, 0x1F, 
    0x10, 0x10, 0x1F, 0x10, 0x1F, 
    0x14, 0x14, 0x14, 0x1F, 0x00, 
    0x10, 0x10, 0x10, 0xF0, 0x00, 
    0x00, 0x00, 0x00, 0x1F, 0x10, 
    0x10, 0x10, 0x10, 0x1F, 0x10, 
    0x10, 0x10, 0x10, 0xF0, 0x10, 
    0x00, 0x00, 0x00, 0xFF, 0x10, 
    0x10, 0x10, 0x10, 0x10, 0x10, 
    0x10, 0x10, 0x10, 0xFF, 0x10, 
    0x00, 0x00, 0x00, 0xFF, 0x14, 
    0x00, 0x00, 0xFF, 0x00, 0xFF, 
    0x00, 0x00, 0x1F, 0x10, 0x17, 
    0x00, 0x00, 0xFC, 0x04, 0xF4, 
    0x14, 0x14, 0x17, 0x10, 0x17, 
    0x14, 0x14, 0xF4, 0x04, 0xF4, 
    0x00, 0x00, 0xFF, 0x00, 0xF7, 
    0x14, 0x14, 0x14, 0x14, 0x14, 
    0x14, 0x14, 0xF7, 0x00, 0xF7, 
    0x14, 0x14, 0x14, 0x17, 0x14, 
    0x10, 0x10, 0x1F, 0x10, 0x1F, 
    0x14, 0x14, 0x14, 0xF4, 0x14, 
    0x10, 0x10, 0xF0, 0x10, 0xF0, 
    0x00, 0x00, 0x1F, 0x10, 0x1F, 
    0x00, 0x00, 0x00, 0x1F, 0x14, 
    0x00, 0x00, 0x00, 0xFC, 0x14, 
    0x00, 0x00, 0xF0, 0x10, 0xF0, 
    0x10, 0x10, 0xFF, 0x10, 0xFF, 
    0x14, 0x14, 0x14, 0xFF, 0x14, 
    0x10, 0x10, 0x10, 0x1F, 0x00, 
    0x00, 0x00, 0x00, 0xF0, 0x10, 
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 
    0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 
    0xFF, 0xFF, 0xFF, 0x00, 0x00, 
    0x00, 0x00, 0x00, 0xFF, 0xFF, 
    0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 
    0x38, 0x44, 0x44, 0x38, 0x44, 
    0x7C, 0x2A, 0x2A, 0x3E, 0x14, 
    0x7E, 0x02, 0x02, 0x06, 0x06, 
    0x02, 0x7E, 0x02, 0x7E, 0x02, 
    0x63, 0x55, 0x49, 0x41, 0x63, 
    0x38, 0x44, 0x44, 0x3C, 0x04, 
    0x40, 0x7E, 0x20, 0x1E, 0x20, 
    0x06, 0x02, 0x7E, 0x02, 0x02, 
    0x99, 0xA5, 0xE7, 0xA5, 0x99, 
    0x1C, 0x2A, 0x49, 0x2A, 0x1C, 
    0x4C, 0x72, 0x01, 0x72, 0x4C, 
    0x30, 0x4A, 0x4D, 0x4D, 0x30, 
    0x30, 0x48, 0x78, 0x48, 0x30, 
    0xBC, 0x62, 0x5A, 0x46, 0x3D, 
    0x3E, 0x49, 0x49, 0x49, 0x00, 
    0x7E, 0x01, 0x01, 0x01, 0x7E, 
    0x2A, 0x2A, 0x2A, 0x2A, 0x2A, 
    0x44, 0x44, 0x5F, 0x44, 0x44, 
    0x40, 0x51, 0x4A, 0x44, 0x40, 
    0x40, 0x44, 0x4A, 0x51, 0x40, 
    0x00, 0x00, 0xFF, 0x01, 0x03, 
    0xE0, 0x80, 0xFF, 0x00, 0x00, 
    0x08, 0x08, 0x6B, 0x6B, 0x08,
    0x36, 0x12, 0x36, 0x24, 0x36, 
    0x06, 0x0F, 0x09, 0x0F, 0x06, 
    0x00, 0x00, 0x18, 0x18, 0x00, 
    0x00, 0x00, 0x10, 0x10, 0x00, 
    0x30, 0x40, 0xFF, 0x01, 0x01, 
    0x00, 0x1F, 0x01, 0x01, 0x1E, 
    0x00, 0x19, 0x1D, 0x17, 0x12, 
    0x00, 0x3C, 0x3C, 0x3C, 0x3C, 
    0x00, 0x00, 0x00, 0x00, 0x00, 
]

OLED_SEEDFONT =[
    [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00],
    [0x00,0x00,0x5F,0x00,0x00,0x00,0x00,0x00],
    [0x00,0x00,0x07,0x00,0x07,0x00,0x00,0x00],
    [0x00,0x14,0x7F,0x14,0x7F,0x14,0x00,0x00],
    [0x00,0x24,0x2A,0x7F,0x2A,0x12,0x00,0x00],
    [0x00,0x23,0x13,0x08,0x64,0x62,0x00,0x00],
    [0x00,0x36,0x49,0x55,0x22,0x50,0x00,0x00],
    [0x00,0x00,0x05,0x03,0x00,0x00,0x00,0x00],
    [0x00,0x1C,0x22,0x41,0x00,0x00,0x00,0x00],
    [0x00,0x41,0x22,0x1C,0x00,0x00,0x00,0x00],
    [0x00,0x08,0x2A,0x1C,0x2A,0x08,0x00,0x00],
    [0x00,0x08,0x08,0x3E,0x08,0x08,0x00,0x00],
    [0x00,0xA0,0x60,0x00,0x00,0x00,0x00,0x00],
    [0x00,0x08,0x08,0x08,0x08,0x08,0x00,0x00],
    [0x00,0x60,0x60,0x00,0x00,0x00,0x00,0x00],
    [0x00,0x20,0x10,0x08,0x04,0x02,0x00,0x00],
    [0x00,0x3E,0x51,0x49,0x45,0x3E,0x00,0x00],
    [0x00,0x00,0x42,0x7F,0x40,0x00,0x00,0x00],
    [0x00,0x62,0x51,0x49,0x49,0x46,0x00,0x00],
    [0x00,0x22,0x41,0x49,0x49,0x36,0x00,0x00],
    [0x00,0x18,0x14,0x12,0x7F,0x10,0x00,0x00],
    [0x00,0x27,0x45,0x45,0x45,0x39,0x00,0x00],
    [0x00,0x3C,0x4A,0x49,0x49,0x30,0x00,0x00],
    [0x00,0x01,0x71,0x09,0x05,0x03,0x00,0x00],
    [0x00,0x36,0x49,0x49,0x49,0x36,0x00,0x00],
    [0x00,0x06,0x49,0x49,0x29,0x1E,0x00,0x00],
    [0x00,0x00,0x36,0x36,0x00,0x00,0x00,0x00],
    [0x00,0x00,0xAC,0x6C,0x00,0x00,0x00,0x00],
    [0x00,0x08,0x14,0x22,0x41,0x00,0x00,0x00],
    [0x00,0x14,0x14,0x14,0x14,0x14,0x00,0x00],
    [0x00,0x41,0x22,0x14,0x08,0x00,0x00,0x00],
    [0x00,0x02,0x01,0x51,0x09,0x06,0x00,0x00],
    [0x00,0x32,0x49,0x79,0x41,0x3E,0x00,0x00],
    [0x00,0x7E,0x09,0x09,0x09,0x7E,0x00,0x00],
    [0x00,0x7F,0x49,0x49,0x49,0x36,0x00,0x00],
    [0x00,0x3E,0x41,0x41,0x41,0x22,0x00,0x00],
    [0x00,0x7F,0x41,0x41,0x22,0x1C,0x00,0x00],
    [0x00,0x7F,0x49,0x49,0x49,0x41,0x00,0x00],
    [0x00,0x7F,0x09,0x09,0x09,0x01,0x00,0x00],
    [0x00,0x3E,0x41,0x41,0x51,0x72,0x00,0x00],
    [0x00,0x7F,0x08,0x08,0x08,0x7F,0x00,0x00],
    [0x00,0x41,0x7F,0x41,0x00,0x00,0x00,0x00],
    [0x00,0x20,0x40,0x41,0x3F,0x01,0x00,0x00],
    [0x00,0x7F,0x08,0x14,0x22,0x41,0x00,0x00],
    [0x00,0x7F,0x40,0x40,0x40,0x40,0x00,0x00],
    [0x00,0x7F,0x02,0x0C,0x02,0x7F,0x00,0x00],
    [0x00,0x7F,0x04,0x08,0x10,0x7F,0x00,0x00],
    [0x00,0x3E,0x41,0x41,0x41,0x3E,0x00,0x00],
    [0x00,0x7F,0x09,0x09,0x09,0x06,0x00,0x00],
    [0x00,0x3E,0x41,0x51,0x21,0x5E,0x00,0x00],
    [0x00,0x7F,0x09,0x19,0x29,0x46,0x00,0x00],
    [0x00,0x26,0x49,0x49,0x49,0x32,0x00,0x00],
    [0x00,0x01,0x01,0x7F,0x01,0x01,0x00,0x00],
    [0x00,0x3F,0x40,0x40,0x40,0x3F,0x00,0x00],
    [0x00,0x1F,0x20,0x40,0x20,0x1F,0x00,0x00],
    [0x00,0x3F,0x40,0x38,0x40,0x3F,0x00,0x00],
    [0x00,0x63,0x14,0x08,0x14,0x63,0x00,0x00],
    [0x00,0x03,0x04,0x78,0x04,0x03,0x00,0x00],
    [0x00,0x61,0x51,0x49,0x45,0x43,0x00,0x00],
    [0x00,0x7F,0x41,0x41,0x00,0x00,0x00,0x00],
    [0x00,0x02,0x04,0x08,0x10,0x20,0x00,0x00],
    [0x00,0x41,0x41,0x7F,0x00,0x00,0x00,0x00],
    [0x00,0x04,0x02,0x01,0x02,0x04,0x00,0x00],
    [0x00,0x80,0x80,0x80,0x80,0x80,0x00,0x00],
    [0x00,0x01,0x02,0x04,0x00,0x00,0x00,0x00],
    [0x00,0x20,0x54,0x54,0x54,0x78,0x00,0x00],
    [0x00,0x7F,0x48,0x44,0x44,0x38,0x00,0x00],
    [0x00,0x38,0x44,0x44,0x28,0x00,0x00,0x00],
    [0x00,0x38,0x44,0x44,0x48,0x7F,0x00,0x00],
    [0x00,0x38,0x54,0x54,0x54,0x18,0x00,0x00],
    [0x00,0x08,0x7E,0x09,0x02,0x00,0x00,0x00],
    [0x00,0x18,0xA4,0xA4,0xA4,0x7C,0x00,0x00],
    [0x00,0x7F,0x08,0x04,0x04,0x78,0x00,0x00],
    [0x00,0x00,0x7D,0x00,0x00,0x00,0x00,0x00],
    [0x00,0x80,0x84,0x7D,0x00,0x00,0x00,0x00],
    [0x00,0x7F,0x10,0x28,0x44,0x00,0x00,0x00],
    [0x00,0x41,0x7F,0x40,0x00,0x00,0x00,0x00],
    [0x00,0x7C,0x04,0x18,0x04,0x78,0x00,0x00],
    [0x00,0x7C,0x08,0x04,0x7C,0x00,0x00,0x00],
    [0x00,0x38,0x44,0x44,0x38,0x00,0x00,0x00],
    [0x00,0xFC,0x24,0x24,0x18,0x00,0x00,0x00],
    [0x00,0x18,0x24,0x24,0xFC,0x00,0x00,0x00],
    [0x00,0x00,0x7C,0x08,0x04,0x00,0x00,0x00],
    [0x00,0x48,0x54,0x54,0x24,0x00,0x00,0x00],
    [0x00,0x04,0x7F,0x44,0x00,0x00,0x00,0x00],
    [0x00,0x3C,0x40,0x40,0x7C,0x00,0x00,0x00],
    [0x00,0x1C,0x20,0x40,0x20,0x1C,0x00,0x00],
    [0x00,0x3C,0x40,0x30,0x40,0x3C,0x00,0x00],
    [0x00,0x44,0x28,0x10,0x28,0x44,0x00,0x00],
    [0x00,0x1C,0xA0,0xA0,0x7C,0x00,0x00,0x00],
    [0x00,0x44,0x64,0x54,0x4C,0x44,0x00,0x00],
    [0x00,0x08,0x36,0x41,0x00,0x00,0x00,0x00],
    [0x00,0x00,0x7F,0x00,0x00,0x00,0x00,0x00],
    [0x00,0x41,0x36,0x08,0x00,0x00,0x00,0x00],
    [0x00,0x02,0x01,0x01,0x02,0x01,0x00,0x00],
    [0x00,0x02,0x05,0x05,0x02,0x00,0x00,0x00] 
]

##################################################################
class Oled(I2c):
    OLED_SSD1306_I2C_128x32 = 0
    OLED_SH1106_I2C_128x64  = 1

    SSD1306_I2C_ADDRESS = 0x3C
    SH1106_I2C_ADDRESS = 0x3C

    BLACK = 0
    WHITE = 1
    
    def __init__(self, addr=SSD1306_I2C_ADDRESS, type=OLED_SSD1306_I2C_128x32, automode=True):
        super().__init__(addr)
        self.oled_type = None

        self.oled_width = 0
        self.oled_height = 0
        self.oled_buff_size = 0
        self.poledbuff = []

        self.rotation = 0
        self.cursor_y = 0
        self.cursor_x = 0
        self.textsize = 1
        self.textcolor = 0xFFFF
        self.textbgcolor = 0xFFFF
        self.wrap = True

        self.automode = automode

        self.init(type)
        

    def __del__(self):
        #self.clearDisplay()
        #self.display()
        self.close()

    def init(self, OLED_TYPE):
        if not self.select_oled(OLED_TYPE):
            return False

        if self.oled_height == 32:
            multiplex = 0x1F
            compins   = 0x02
            #compins   = 0x20
            #compins   = 0x00
            contrast  = 0x8F
        else:
            multiplex = 0x3F
            compins   = 0x12
            if oled_type == OLED_SH1106_I2C_128x64:
                contrast = 0x80
            else:
                contrast = 0x9F if self.vcc_type == SSD_External_Vcc else 0xCF

        if self.vcc_type == SSD_External_Vcc:
            chargepump = 0x10 
            precharge  = 0x22
        else:
            chargepump = 0x14 
            precharge  = 0xF1

        self.sendCommand(SSD_Display_Off)                    
        self.sendCommand(SSD_Set_Muliplex_Ratio, multiplex)
        time.sleep(0.0001)

        if self.oled_type == OLED_SH1106_I2C_128x64:
            self.sendCommand(SSD1306_Set_Lower_Column_Start_Address|0x02) 
            self.sendCommand(SSD1306_Set_Higher_Column_Start_Address) 
            self.sendCommand(SSD1306_Set_Start_Line)     
            self.sendCommand(SH1106_Set_Page_Address)    
            self.sendCommand(SSD_Set_Segment_Remap|0x01) 
            self.sendCommand(SSD1306_Normal_Display)
            self.sendCommand(0xad)    
            self.sendCommand(0x8b)    
            self.sendCommand(0x30)    
            self.sendCommand(SSD1306_Set_Com_Output_Scan_Direction_Remap)    
            self.sendCommand(SSD1306_Set_Display_Offset)    
            self.sendCommand(0x00)   
            self.sendCommand(SSD1306_Set_Display_Clock_Div)   
            self.sendCommand(0x80)
            self.sendCommand(SSD1306_Set_Precharge_Period)    
            self.sendCommand(0x1f)    
            self.sendCommand(SSD1306_Set_Com_Pins)    
            self.sendCommand(0x12)
            self.sendCommand(SSD1306_Set_Vcomh_Deselect_Level)    
            self.sendCommand(0x40)
        else:
            self.sendCommand(SSD1306_Charge_Pump_Setting, chargepump) 
            self.sendCommand(SSD1306_Set_Memory_Mode, 0x00)           
            self.sendCommand(SSD1306_Set_Display_Clock_Div, 0x80)     
            self.sendCommand(SSD1306_Set_Display_Offset, 0x00)        
            self.sendCommand(SSD1306_Set_Start_Line | 0x0)            
            self.sendCommand(SSD_Set_Segment_Remap | 0x1)
            self.sendCommand(SSD1306_Set_Com_Output_Scan_Direction_Remap)

            self.sendCommand(SSD1306_Set_Com_Pins, compins)  
            self.sendCommand(SSD1306_Set_Precharge_Period, precharge) 
            self.sendCommand(SSD1306_Set_Vcomh_Deselect_Level, 0x40) 
            self.sendCommand(SSD1306_Entire_Display_Resume)    
            self.sendCommand(SSD1306_Normal_Display)         
            self.sendCommand(SSD_Set_Column_Address,0,127) 
            self.sendCommand(SSD_Set_Page_Address, 0,7) 

        self.sendCommand(SSD_Set_ContrastLevel, contrast)
        self.stopscroll()

        self.setTextSize(1)
        self.setTextColor(WHITE)
        self.clearDisplay()
        self.sendCommand(SSD_Display_On)              
        time.sleep(0.1)

        return True

    def I2CWrite(self, tbuf, length):
        if length==2:
            I2c.writeByte(self, tbuf[0],tbuf[1])
        elif length==3:
            I2c.writeWord(self, tbuf[0], (tbuf[2]<<8) | tbuf[1])
        else:
            I2c.writeBlock(self, tbuf[0], tbuf[1:])

    def print(self, string):
        for p in string:
            if p == '\n':
                self.cursor_y += self.textsize * 8
                self.cursor_x = 0
            elif p == '\r':
                pass
            else:
                self.drawChar(self.cursor_x, self.cursor_y, p, self.textcolor, self.textbgcolor, self.textsize)
                self.cursor_x += self.textsize*6

                if self.wrap and (self.cursor_x > (self.oled_width - self.textsize * 6)):
                    self.cursor_y += self.textsize*8
                    self.cursor_x = 0
                    
        if self.automode:
            self.display()

    def drawCircle(self, x0, y0, r, color):
        f = 1 - r
        ddF_x = 1
        ddF_y = -2 * r
        x = 0
        y = r

        self._drawPixel(x0, y0+r, color)
        self._drawPixel(x0, y0-r, color)
        self._drawPixel(x0+r, y0, color)
        self._drawPixel(x0-r, y0, color)

        while x < y:
            if f >= 0:
                y -= 1
                ddF_y += 2
                f += ddF_y

            x += 1
            ddF_x += 2
            f += ddF_x
            self._drawPixel(x0 + x, y0 + y, color)
            self._drawPixel(x0 - x, y0 + y, color)
            self._drawPixel(x0 + x, y0 - y, color)
            self._drawPixel(x0 - x, y0 - y, color)
            self._drawPixel(x0 + y, y0 + x, color)
            self._drawPixel(x0 - y, y0 + x, color)
            self._drawPixel(x0 + y, y0 - x, color)
            self._drawPixel(x0 - y, y0 - x, color)

        if self.automode:
            self.display()

    def drawCircleHelper(self, x0, y0, r, cornername, color):
        f = 1 - r
        ddF_x = 1
        ddF_y = -2 * r
        x = 0
        y = r

        while x < y:
            if f >= 0:
                y -= 1
                ddF_y += 2
                f += ddF_y

            x += 1
            ddF_x += 2
            f += ddF_x
            if cornername & 0x4:
                self._drawPixel(x0 + x, y0 + y, color)
                self._drawPixel(x0 + y, y0 + x, color)

            if cornername & 0x2:
                self._drawPixel(x0 + x, y0 - y, color)
                self._drawPixel(x0 + y, y0 - x, color)

            if cornername & 0x8:
                self._drawPixel(x0 - y, y0 + x, color)
                self._drawPixel(x0 - x, y0 + y, color)

            if cornername & 0x1:
                self._drawPixel(x0 - y, y0 - x, color)
                self._drawPixel(x0 - x, y0 - y, color)

        if self.automode:
            self.display()

    def fillCircle(self, x0, y0, r, color):
        self._drawFastVLine(x0, y0-r, 2*r+1, color)
        self.fillCircleHelper(x0, y0, r, 3, 0, color)

        if self.automode:
            self.display()

    def fillCircleHelper(self, x0, y0, r, cornername, delta, color):
        f = 1 - r
        ddF_x = 1
        ddF_y = -2 * r
        x = 0
        y = r

        while x < y:
            if f >= 0:
                y -= 1
                ddF_y += 2
                f += ddF_y

            x += 1
            ddF_x += 2
            f += ddF_x

            if cornername & 0x1:
                self._drawFastVLine(x0+x, y0-y, 2*y+1+delta, color)
                self._drawFastVLine(x0+y, y0-x, 2*x+1+delta, color)

            if cornername & 0x2:
                self._drawFastVLine(x0-x, y0-y, 2*y+1+delta, color)
                self._drawFastVLine(x0-y, y0-x, 2*x+1+delta, color)
                
        if self.automode:
            self.display()

        
    def drawLine(self, x0, y0, x1, y1, color):
        self._drawLine(x0, y0, x1, y1, color)

        if self.automode:
            self.display()

    def _drawLine(self, x0, y0, x1, y1, color):
        steep = abs(y1 - y0) > abs(x1 - x0)

        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1

        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        dx = x1 - x0
        dy = abs(y1 - y0)
        err = dx / 2

        if y0 < y1:
            ystep = 1
        else:
            ystep = -1


        while x0 <= x1:
            if steep:
                self._drawPixel(y0, x0, color)
            else:
                self._drawPixel(x0, y0, color)

            err -= dy

            if err < 0:
                y0 += ystep
                err += dx

            x0 += 1

    def drawRect(self, x, y, w, h, color):
        self._drawFastHLine(x, y, w, color)
        self._drawFastHLine(x, y+h-1, w, color)
        self._drawFastVLine(x, y, h, color)
        self._drawFastVLine(x+w-1, y, h, color)

        if self.automode:
            self.display()

    def drawFastVLine(self, x, y, h, color):
        self._drawFastVLine(x, y, h, color)

        if self.automode:
            self.display()

    def _drawFastVLine(self, x, y, h, color):
        self._drawLine(x, y, x, y+h-1, color)

    def drawFastHLine(self, x, y, w, color):
        self._drawFastHLine(x, y, w, color)

        if self.automode:
            self.display()

    def _drawFastHLine(self, x, y, w, color):
        self._drawLine(x, y, x+w-1, y, color)

    def fillRect(self, x, y, w, h, color):
        self._fillRect(x, y, w, h, color)

        if self.automode:
            self.display()

    def _fillRect(self, x, y, w, h, color):
        for i in range(x, x+w):
            self._drawFastVLine(i, y, h, color) 

    def drawVerticalBargraph(self, x, y, w, h, color, percent) :
        self.drawRect(x, y, w, h, color)
        if h>2 and w>2:
            vsize = int(((h-2)*percent)/100)
            self._fillRect(x+1, y+1+((h-2)-vsize), w - 2, vsize, color)

        if self.automode:
            self.display()

    def drawHorizontalBargraph(self, x, y, w, h, color, percent):
        self.drawRect(x, y, w, h, color)
        if h>2 and w>2:
            hsize = int(((w - 2) * percent) / 100)
            self._fillRect(x+1, y+1, hsize, h-2, color)

        if self.automode:
            self.display()

    def fillScreen(self, color):
        self._fillRect(0, 0, oled_width, oled_height, color)

        if self.automode:
            self.display()

    def drawRoundRect(self, x, y, w, h, r, color):
        self._drawFastHLine(x+r,y,w-2*r,color)
        self._drawFastHLine(x+r,y+h-1,w-2*r,color)
        self._drawFastVLine(x,y+r,h-2*r,color)
        self._drawFastVLine(x+w-1,y+r,h-2*r,color)

        self.drawCircleHelper(x+r,y+r,r,1,color)
        self.drawCircleHelper(x+w-r-1,y+r,r,2,color)
        self.drawCircleHelper(x+w-r-1,y+h-r-1,r,4,color)
        self.drawCircleHelper(x+r,y+h-r-1,r,8,color)

        if self.automode:
            self.display()

    def fillRoundRect(self, x, y, w, h, r, color):
        self._fillRect(x+r,y,w-2*r,h,color)
        self.fillCircleHelper(x+w-r-1,y+r,r,1,h-2*r-1,color)
        self.fillCircleHelper(x+r,y+r,r,2,h-2*r-1,color)

        if self.automode:
            self.display()

    def drawTriangle(self, x0, y0, x1, y1, x2, y2, color):
        self._drawLine(x0, y0, x1, y1, color)
        self._drawLine(x1, y1, x2, y2, color)
        self._drawLine(x2, y2, x0, y0, color)

        if self.automode:
            self.display()

    def fillTriangle(self, x0, y0, x1, y1, x2, y2, color):
        dx01 = x1 - x0
        dy01 = y1 - y0
        dx02 = x2 - x0
        dy02 = y2 - y0
        dx12 = x2 - x1
        dy12 = y2 - y1
        sa = 0
        sb = 0

        if y0 > y1:
            y0, y1 = y1, y0 
            x0, x1 = x1, x0

        if y1 > y2:
            y2, y1 = y1, y2
            x2, x1 = x1, x2

        if y0 > y1:
            y0, y1 = y1, y0
            x0, x1 = x1, x0

        if y0 == y2: 
            a = b = x0
            if x1 < a:
                a = x1
            elif x1 > b:
                b = x1
            if x2 < a:     
                a = x2
            elif x2 > b:
                b = x2
            self._drawFastHLine(a, y0, b-a+1, color)
            return

        if y1 == y2:
            last = y1   
        else:
            last = y1-1 

        for y in range(y0, last+1):
            a = x0 + sa / dy01
            b = x0 + sb / dy02
            sa += dx01
            sb += dx02
            if a > b:
                a, b = b, a
            a = int(a)
            b = int(b)
            self._drawFastHLine(a, y, b-a+1, color)

        sa = dx12 * (y - y1)
        sb = dx02 * (y - y0)

        for y in range(last+1, y2+1):
            a = x1 + sa / dy12
            b = x0 + sb / dy02
            sa += dx12
            sb += dx02
            if a > b:
                a, b = b, a
            a = int(a)
            b = int(b)
            self._drawFastHLine(a, y, b-a+1, color)

        if self.automode:
            self.display()

    def drawBitmap(self, x, y, bitmap, w, h, color):
        byteWidth = int((w + 7) / 8)

        for j in range(0 ,h):
            for i in range(0 ,w):
                if (bitmap[int(j * byteWidth + i / 8)]) & (128 >> (i & 7)):
                    self._drawPixel(x+i, y+j, color)

        if self.automode:
            self.display()

    def drawChar(self, x, y, c, color, bg, size):
        if (x >= self.oled_width) or (y >= self.oled_height) or ((x + 5 * size - 1) < 0) or((y + 8 * size - 1) < 0):
            return

        for i in range(6):
            if i == 5:
                line = 0x0
            else:
                if type(c) is int:
                    line = OLED_FONT[(c*5)+i]
                else:
                    line = OLED_FONT[(ord(c)*5)+i]

            for j in range(8):
                if line & 0x1:
                    if size == 1:
                        self._drawPixel(x+i, y+j, color)
                    else:  
                        self._fillRect(x+(i*size), y+(j*size), size, size, color)

                elif bg != color:
                    if size == 1: 
                        self._drawPixel(x+i, y+j, bg)
                    else:
                        self._fillRect(x+i*size, y+j*size, size, size, bg)

                line >>= 1        

        if self.automode:
            self.display()

    def setCursor(self, x, y):
        self.cursor_x = x
        self.cursor_y = y

        if self.automode:
            self.display()

    def setTextSize(self, s):
        self.textsize = s if s > 0 else 1

        if self.automode:
            self.display()

    def setTextColor(self, c, b=None):
        self.textcolor = c
        self.textbgcolor = c if b is None else b 

        if self.automode:
            self.display()

    def setTextWrap(self, w):
        self.wrap = w

        if self.automode:
            self.display()

    def width(self): 
        return self.oled_width 

    def height(self): 
        return self.oled_height 

    def drawPixel(self, x, y, color):
        self._drawPixel(x, y, color)

        if self.automode:
            self.display()

    def _drawPixel(self, x, y, color):
        if not(type(x) is int) or not(type(y) is int) :
            raise TypeError("x and y must be integer x: {}/y: {}".format(x, y))
        if (x < 0) or (x >= self.width()) or (y < 0) or (y >= self.height()):
            return

        if color == WHITE:
            self.poledbuff[round(x + int(y/8) * self.oled_width)] |= (1 << (y%8))
        else:
            self.poledbuff[round(x + int(y/8) * self.oled_width)] &= ~(1 << (y%8))

    def select_oled(self, OLED_TYPE):
        self.oled_width  = 128
        self.oled_type = OLED_TYPE
        self.vcc_type = SSD_Internal_Vcc
        if OLED_TYPE == OLED_SSD1306_I2C_128x32:
            self.oled_height = 32
        elif OLED_TYPE ==  OLED_SH1106_I2C_128x64:
            self.oled_height = 64
        else:
            return False

        self.oled_buff_size = int(self.oled_width * self.oled_height / 8) * 2
        self.poledbuff = [0] * self.oled_buff_size
        return True

    def close(self):
        self.poledbuff = None

    def setSeedTextXY(self, Row, Column):
        self.sendCommand(0x15)             
        self.sendCommand(0x08+(Column*4))  
        self.sendCommand(0x37)             
        self.sendCommand(0x75)             
        self.sendCommand(0x00+(Row*8))     
        self.sendCommand(0x07+(Row*8))

    def putSeedChar(self, C):
        if C < 32 or C > 127:
            C=' ' #Space

        for i in range(0, 8, 2):
            for j in range(0, 8):
                c = 0x00
                bit1 = (OLED_SEEDFONT[C-32][i]   >> j) & 0x01  
                bit2 = (OLED_SEEDFONT[C-32][i+1] >> j) & 0x01

                if bit1:
                    c |= grayH
                if bit2:
                    c |= grayL

                self.sendData(c)

    def putSeedString(self, string):
        for s in string:
            putSeedChar(s)

    def setBrightness(self, Brightness):
        self.sendCommand(SSD_Set_ContrastLevel)
        self.sendCommand(Brightness)

        if self.automode:
            self.display()

    def invertDisplay(self, i):
        if i:
            self.sendCommand(SSD_Inverse_Display)
        else:
            self.sendCommand(SSD1306_Normal_Display)

        if self.automode:
            self.display()

    def sendCommand(self, c0, c1=None, c2=None):
        if c1 is None:
            buff = [0] * 2
            buff[0] = SSD_Command_Mode 
            buff[1] = c0
            self.I2CWrite(buff, len(buff))
        elif c2 is None:
            buff = [0] * 3
            buff[0] = SSD_Command_Mode
            buff[1] = c0
            buff[2] = c1
            self.I2CWrite(buff, len(buff))
        else:
            buff = [0] * 4
            buff[0] = SSD_Command_Mode 
            buff[1] = c0
            buff[2] = c1
            buff[3] = c2
        self.I2CWrite(buff, len(buff))

    def startscrollright(self, start, stop):
        self.sendCommand(SSD_Right_Horizontal_Scroll)
        self.sendCommand(0x00)
        self.sendCommand(start)
        self.sendCommand(0x00)
        self.sendCommand(stop)
        self.sendCommand(0x01)
        self.sendCommand(0xFF)
        self.sendCommand(SSD_Activate_Scroll)

    def startscrollleft(self, start, stop):
        self.sendCommand(SSD_Left_Horizontal_Scroll)
        self.sendCommand(0x00)
        self.sendCommand(start)
        self.sendCommand(0x00)
        self.sendCommand(stop)
        self.sendCommand(0x01)
        self.sendCommand(0xFF)
        self.sendCommand(SSD_Activate_Scroll)

    def startscrolldiagright(self, start, stop):
        self.sendCommand(SSD1306_SET_VERTICAL_SCROLL_AREA)  
        self.sendCommand(0x00)
        self.sendCommand(self.oled_height)
        self.sendCommand(SSD1306_VERTICAL_AND_RIGHT_HORIZONTAL_SCROLL)
        self.sendCommand(0x00)
        self.sendCommand(start)
        self.sendCommand(0x00)
        self.sendCommand(stop)
        self.sendCommand(0x01)
        self.sendCommand(SSD_Activate_Scroll)

    def startscrolldiagleft(self, start, stop):
        self.sendCommand(SSD1306_SET_VERTICAL_SCROLL_AREA)  
        self.sendCommand(0x00)
        self.sendCommand(self.oled_height)
        self.sendCommand(SSD1306_VERTICAL_AND_LEFT_HORIZONTAL_SCROLL)
        self.sendCommand(0x00)
        self.sendCommand(start)
        self.sendCommand(0x00)
        self.sendCommand(stop)
        self.sendCommand(0x01)
        self.sendCommand(SSD_Activate_Scroll)

    def setHorizontalScrollProperties(self, direction, startRow, endRow, startColumn, endColumn, scrollSpeed):
        if Scroll_Right == direction:
            self.sendCommand(SSD_Left_Horizontal_Scroll)
        else:
            self.sendCommand(SSD_Right_Horizontal_Scroll)

        self.sendCommand(0x00)       
        self.sendCommand(startRow)
        self.sendCommand(scrollSpeed)
        self.sendCommand(endRow)
        self.sendCommand(startColumn+8)
        self.sendCommand(endColumn+8)
        self.sendCommand(0x00)

    def stopscroll(self):
        self.sendCommand(SSD_Deactivate_Scroll)

    def sendData(self, c):
        buff = [0] * 2
        buff[0] = SSD_Data_Mode 
        buff[1] = c
        self.I2CWrite(buff, len(buff))

    def display(self):
        pos = -1
        line = 0
        buff = [0] * 17
        
        self.sendCommand(SSD1306_Set_Lower_Column_Start_Address | 0x0) 
        self.sendCommand(SSD1306_Set_Higher_Column_Start_Address | 0x0) 
        self.sendCommand(SSD1306_Set_Start_Line | 0x0) 

        buff[0] = SSD_Data_Mode

        if self.oled_type == OLED_SH1106_I2C_128x64:
            for k in range(0, 8):
                self.sendCommand(0xB0+k)
                self.sendCommand(0x02) 
                self.sendCommand(0x10) 

                for i in range(0, 8):
                    for x in range(1,  16 + 1): 
                        buff[x] = self.poledbuff[pos+x]
                    self.I2CWrite(buff, len(buff))            
                    pos += 16
        else:
            for i in range(0, self.oled_buff_size, 16):
                for x in range(1, 16+1):
                    buff[x] = self.poledbuff[pos+x]
                self.I2CWrite(buff, 17)
                pos += 16

    def clearDisplay(self):
        self.poledbuff = [0] * self.oled_buff_size

        if self.automode:
            self.display()

    def write(self, c) :
        if c == '\n':
            self.cursor_y += self.textsize*8
            self.cursor_x = 0
        elif c == '\r':
            pass
        else:
            self.drawChar(self.cursor_x, self.cursor_y, c, self.textcolor, self.textbgcolor, self.textsize)
            self.cursor_x += self.textsize*6

            if self.wrap and (self.cursor_x > (self.oled_width - self.textsize*6)):
                self.cursor_y += self.textsize*8
                self.cursor_x = 0

        if self.automode:
            self.display()

    def setAutomode(self, automode):
        self.automode = automode

##################################################################
class PixelDisplay():
    #Integer Color Value
    RED     = 0x00200000
    ORANGE  = 0x00201000
    YELLOW  = 0x00202000
    GREEN   = 0x00002000
    SKYBLUE = 0x00002020
    BLUE    = 0x00000020
    PURPLE  = 0x00100010
    PINK    = 0x0046000E
    WHITE   = 0x00101010

    def __init__(self, width=8, height=8, *, type="GRB", automode=True, bus=0, device=0):
        self.type = type
        self.width = width
        self.height = height
        self.automode = automode
        self.brightness = 0

        self.invert = False
        self._buf = [[0 for __ in range(self.width)] for __ in range(self.height)]        
        self._neopixel = NeoPixel_SPI(spi=board.SPI(), n=width * height, pixel_order=type, auto_write=False)
        self.setBrightness(50)

        self.clear()
    
    def __del__(self):
        self.clear()

    def __enter__(self, width=8, height=8, *, type="GRB", automode=True, bus=0, device=0):
        self.__init__(width=8, height=8, type="GRB", automode=True, bus=0, device=0)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.clear()

    def fill(self, color):
        color = self.RGBtoHEX(color)        
        if self.invert:
            color = ~color & 0xFFFFFF

        self._buf = [[color for __ in range(self.width)] for __ in range(self.height)]
        self.display()
    
    def clear(self):
        self._buf = [[0 for __ in range(self.width)] for __ in range(self.height)]
        self.display()

    def setColor(self, x, y, color):
        over_width = x > self.width
        over_height = y > self.height

        if over_width or over_height:
            raise IndexError
        
        if self.invert:
            color = ~color & 0xFFFFFF

        self._buf[y][x] = color

        if self.automode:
            self.display()

    def getColor(self, x, y):
        color = self.HEXtoRGB(self._buf[y][x])
        return color

    def setAutomode(self, automode):
        self._neopixel.auto_write = automode
    
    def rainbow(self, duration=5, delay=0.05):
        import copy

        colors = [
                self.RED, 
                self.ORANGE, 
                self.YELLOW, 
                self.GREEN, 
                self.SKYBLUE, 
                self.BLUE, 
                self.PURPLE, 
                self.PINK]

        loop = int(duration / delay)
        for __ in range(loop):
            self._buf[1:self.height] = copy.deepcopy(self._buf[:self.height-1])

            for x in range(self.width):
                self._buf[0][x] = colors[x % len(colors)]
            colors = colors[1:] + colors[:1]

            self.display()
            time.sleep(delay)

    def setColorInvert(self, invert):
        self.invert = invert

    def setBrightness(self, brightness):
        self.brightness = brightness
        self._neopixel.brightness = brightness / 255

    def display(self):
        for y in range(self.height):
            for x in range(self.width):
                if x % 2:
                    self._neopixel[(x + 1) * self.height - y - 1] = self._buf[y][x]
                else:
                    self._neopixel[x * self.height + y] = self._buf[y][x]
        self._neopixel.show()

    def getRGBType(self):
        return self.type

    def RGBtoHEX(self, color_arr):
        color = 0
        _type = type(color_arr)

        if _type is list or _type is tuple:
            color_arr = list(color_arr)

            for i in range(3):
                color_arr[i] = 0 if color_arr[i] < 0 else color_arr[i]
                color_arr[i] = 255 if color_arr[i] > 255 else color_arr[i]

                color = color + (color_arr[i] << (16 - i * 8))

        elif _type is int:
            color = color_arr

        else:
            raise TypeError("color must be int or array type not {%s}".format(_type))

        return color

    def HEXtoRGB(self, color):
        color_arr = [0] * 3
        _type = type(color)

        if _type is int:
            color_arr = list(color_arr)
            for i in range(3):
                color_arr[i] = (color >> (16 - i * 8)) & 0xFF
            
        else:
            raise TypeError("color must be int not {}".format(_type))

        return color_arr

##################################################################
class ShiftRegister(object):
    _fnd_map = [
        [0,1,1,1,1,1,1,0], #0
        [0,0,0,0,1,1,0,0], #1
        [1,0,1,1,0,1,1,0], #2
        [1,0,0,1,1,1,1,0], #3
        [1,1,0,0,1,1,0,0], #4
        [1,1,0,1,1,0,1,0], #5
        [1,1,1,1,1,0,1,0], #6
        [0,1,0,0,1,1,1,0], #7
        [1,1,1,1,1,1,1,0], #8
        [1,1,0,1,1,1,1,0]  #9
    ]

    def __init__(self, n):
        self._data = n[0]
        self._clock = n[1]        
        self._latch = n[2]
        GPIO.setup(self._data,GPIO.OUT)
        GPIO.setup(self._clock,GPIO.OUT)
        GPIO.setup(self._latch,GPIO.OUT)

    def __del__(self):
        GPIO.output(self._data,GPIO.LOW)
        GPIO.output(self._clock,GPIO.LOW)
        GPIO.output(self._latch,GPIO.LOW)

    def shiftout(self, value):
        for i in range(8):
            GPIO.output(self._latch,GPIO.LOW)
            GPIO.output(self._clock,GPIO.LOW)
            if ((value>>i)&0x1):
                GPIO.output(self._data,GPIO.HIGH)
            else:
                GPIO.output(self._data,GPIO.LOW)
            GPIO.output(self._clock,GPIO.HIGH)
            GPIO.output(self._latch,GPIO.HIGH)
            
    def fnd(self, value):
        for i in range(8):
            GPIO.output(self._latch,GPIO.LOW)
            GPIO.output(self._clock,GPIO.LOW)
            GPIO.output(self._data,self._fnd_map[value][i])
            GPIO.output(self._clock,GPIO.HIGH)
            GPIO.output(self._latch,GPIO.HIGH)

def shiftRegister_unittest():
    gpio = [16,6,5]
    shiftregister = ShiftRegister(gpio)
    for i in range(10):
        shiftregister.shiftout(0x7E)
        shiftregister.shiftout(0x30)
        time.sleep(0.1)
    for i in range(10):
        shiftregister.fnd(i)
        shiftregister.fnd(i)
        time.sleep(0.1)

##################################################################
class PiezoBuzzer(PopThread):
    def __init__(self,n,freq=261,duty=0):
        GPIO.setup(n,GPIO.OUT)
        self.piezo = GPIO.PWM(n,freq)
        self.tempo = 120
        self.piezo.start(duty)

    def __del__(self):
        self.piezo.stop()

    def setFreq(self,freq):
        self.piezo.ChangeFrequency(freq)
    
    def setDuty(self,duty):
        self.piezo.ChangeDutyCycle(duty)

    def tone(self,scale,pitch,duration):

        TERM = 50
        
        freq = int(math.pow(2,scale-1)*55*pow(2,(pitch -10)/12))
        self.setFreq(freq)
        self.piezo.start(50)
        
        loop = (int)((60000.0 / self.tempo) * (1.0 / duration * 4))
        time.sleep((loop - TERM)/1000)
        self.piezo.stop()
                
    def rest(self,duration):
        self.tone(0, 0, duration)
        
    def setTempo(self,tempo):
        self.tempo = tempo
        
    def getTempo():
        return self.tempo

    def play(self, sheet):
        self._Sheet = sheet
        self.start(daemon=False)
    
    def isPlay(self):
        return self.isRun()
    
    def run(self):
        for s, p, d in zip(self._Sheet[0], self._Sheet[1], self._Sheet[2]):
            self.tone(s, p, d)
            if not self.isRun():
                break
            
        self.stop()

##################################################################

def bgr8_to_jpeg(value):
    return bytes(cv2.imencode('.jpg', value)[1])

class _camera(SingletonConfigurable):
    value = traitlets.Any()

    width = traitlets.Integer(default_value=224).tag(config=True)
    height = traitlets.Integer(default_value=224).tag(config=True)
    fps = traitlets.Integer(default_value=21).tag(config=True)
    capture_width = traitlets.Integer(default_value=3280).tag(config=True)
    capture_height = traitlets.Integer(default_value=2464).tag(config=True)

    def __init__(self, *args, **kwargs):
        self.value = np.empty((self.height, self.width, 3), dtype=np.uint8)
        super(_camera, self).__init__(*args, **kwargs)

        try:
            self.cap = cv2.VideoCapture(self._gst_str(), cv2.CAP_GSTREAMER)

            re, image = self.cap.read()

            if not re:
                raise RuntimeError('Could not read image from camera.')

            self.value = image
            self.start()
        except:
            self.stop()
            raise RuntimeError('Could not initialize camera.  Please see error trace.')

        atexit.register(self.stop)

    def _capture_frames(self):
        while True:
            re, image = self.cap.read()
            if re:
                self.value = image
            else:
                break

    def _gst_str(self):
        return 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=%d, height=%d, format=(string)NV12, framerate=(fraction)%d/1 ! nvvidconv flip-method=%s ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! videoconvert ! appsink' % (self.capture_width, self.capture_height, self.fps, __main__._camera_flip_method, self.width, self.height)

    def start(self):
        if not self.cap.isOpened():
            self.cap.open(self._gst_str(), cv2.CAP_GSTREAMER)
        if not hasattr(self, 'thread') or not self.thread.isAlive():
            self.thread = Thread(target=self._capture_frames)
            self.thread.start()

    def stop(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'thread'):
            self.thread.join()

    def restart(self):
        self.stop()
        self.start()

class Camera:
    code="7377EF6E7F5DC659B18B3A089F8BD812"
    camera=None
    camera_link=None
    image=None
    width=None
    height=None

    def __init__(self, width=224, height=224, auto_load=True):
        self.width=width
        self.height=height

        if auto_load:
            self.load()

    def __call__(self):
        self.camera_link.link()
        return self.image

    def load(self):
        os.system("echo soda | sudo -S systemctl restart nvargus-daemon")
        self.camera = _camera.instance(width=self.width, height=self.height)
        self.image = widgets.Image(format='jpeg', width=self.width, height=self.height)
        self.camera_link = traitlets.dlink((self.camera, 'value'), (self.image, 'value'), transform=bgr8_to_jpeg)

    def show(self):
        if self.camera is None:
            self.load()

        if self.camera_link is None:
            self.camera_link = traitlets.dlink((self.camera, 'value'), (self.image, 'value'), transform=bgr8_to_jpeg)
            display(self.image)
        else:
            self.camera_link.link()
            display(self.image)

    def stop(self):
        if self.camera_link is not None:
            self.camera_link.unlink()

    @property
    def value(self):
        return self.camera.value

##################################################################
# Audio classes

class Audio:
    def __init__(self, blocking=True, cont=False):
        self.W = None
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.blocking = blocking
        self.cont = cont
        self.isStop = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def stop(self):
        self.isStop = True

    def close(self):
        self.stop()
        time.sleep(.1)

        if self.W != None:
            if type(self.W) is list:
                for w in self.W:
                    w.close()
            else:
                self.W.close()

        if self.stream != None:
            self.stream.stop_stream()
            self.stream.close()
        
        self.p.terminate()


def audio_play(file):
    w = wave.open(file, 'rb')
    data = w.readframes(w.getnframes())
    w.close()

    p = pyaudio.PyAudio()
    s = p.open(format=p.get_format_from_width(w.getsampwidth()), channels=w.getnchannels(), 
                      rate=w.getframerate(), output=True)
    s.write(data)
    s.stop_stream()
    s.close()
    p.terminate()

##################################################################
# AudioPlay classes

class AudioPlay(Audio):
    def __init__(self, file, blocking=True, cont=False):
        super().__init__(blocking, cont)

        self.W = wave.open(file, "rb")
        self.stream = self.p.open(format=self.p.get_format_from_width(self.W.getsampwidth()),
            channels=self.W.getnchannels(), rate=self.W.getframerate(), output=True, 
            stream_callback=None if blocking else self._callback)

        if blocking:
            self.data = self.W.readframes(self.W.getnframes())

    def __del__(self):
        super().__del__()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def _callback(self, in_data, frame_count, time_info, status):
        data = self.W.readframes(frame_count)
        if self.cont:
            mod = frame_count - len(data) // self.W.getsampwidth()
            if mod != 0:
                self.W.rewind()
                data += self.W.readframes(mod)

        return (data, pyaudio.paContinue if not self.isStop else pyaudio.paAbort)   

    def run(self):
        self.isStop = False
        if self.blocking:
            self.stream.write(self.data)
        else:
            self.stream.start_stream()

    def isPlay(self):
        return self.stream.is_active()

##################################################################
# AudioPlayList classes

class AudioPlayList(Audio):
    def __init__(self, files, blocking=True, cont=False):
        super().__init__(blocking, cont)

        self.W = []
        for file in files:
            self.W.append(wave.open(file, "rb"))

        self.stream = self.p.open(format=self.p.get_format_from_width(self.W[0].getsampwidth()),
            channels=self.W[0].getnchannels(), rate=self.W[0].getframerate(), output=True, 
            stream_callback=None if blocking else self._callback)

        self.data = []
        if blocking:
            for w in self.W:
                self.data.append(w.readframes(w.getnframes()))

    def _callback(self, in_data, frame_count, time_info, status):
        data = self.W[self.pos].readframes(frame_count)
        if self.cont:
            mod = frame_count - len(data) // self.W[self.pos].getsampwidth()
            if mod != 0:
                self.pos += 1
                if self.pos >= len(self.W):
                    self.pos = 0

                self.W[self.pos].rewind()
                data += self.W[self.pos].readframes(mod)

        return (data, pyaudio.paContinue if not self.isStop else pyaudio.paAbort)   

    def run(self, pos=0):
        self.isStop = False
        self.pos = pos

        if self.blocking:
            self.stream.write(self.data[pos])            
        else:
            self.stream.start_stream()
    
    def isPlay(self):
        return self.stream.is_active()

##################################################################
# Audio classes

class AudioRecord(Audio):
    def __init__(self, file, sFormat=8, sChannel=1, sRate=48000, sFramePerBuffer=1024):
        super().__init__(False)

        self.w = wave.open(file, "wb")
        self.w.setsampwidth(self.p.get_sample_size(sFormat))
        self.w.setnchannels(sChannel)
        self.w.setframerate(sRate)

        self.stream = self.p.open(format=sFormat, channels=sChannel, rate=sRate, input=True, 
            frames_per_buffer=sFramePerBuffer, stream_callback=self._callback)

    def __del__(self):
        super().__del__()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def _callback(self, in_data, frame_count, time_info, status):
        self.w.writeframes(in_data)
        data = chr(0) * len(in_data)

        return (data, pyaudio.paContinue if not self.isStop else pyaudio.paAbort)   

    def run(self):
        self.stream.start_stream()


class Tone:
    def __init__(self, tempo=100, volume=.5, rate=48000, channels=1): 
        self.tempo = tempo
        self.volume = volume
        self.rate = rate
        self.channels = channels
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paFloat32, channels=self.channels, rate=self.rate, output=True)

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        self.stream.stop_stream()
        self.stream.close() 
        self.p.terminate()

    def setTempo(self, tempo):
        self.tempo = tempo

    def rest(self, duration):
        self.play(0, "REST", 1/4)

    def play(self, octave, pitch, duration):
        """
        octave = 1 ~ 8
        note = DO, RE, MI, FA, SOL, RA, SI
        dulation = 1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, ...
        """
        string_to_pitch = {"REST":0, "DO":1, "DO#":2, "RE":3, "RE#":4, "MI":5, "FA":6, "FA#":7, "SOL":8, "SOL#":9, "RA":10, "RA#":11, "SI":12}

        p = string_to_pitch[pitch]
        f = 2**(octave) * 55 * 2**((p - 10) / 12)
        
        if p == 0:
            time.sleep((60.0 / self.tempo) * (duration * 4))
        else:
            sample = (np.sin(2 * np.pi * np.arange(self.rate * (60.0 / self.tempo * 4) * (duration*4)) * f / self.rate)).astype(np.float32)
            self.stream.write(self.volume * sample)

            time.sleep(0.02)
##################################################################

class Bme680(BME680):
    # Heater control settings
    ENABLE_HEATER = 0x00
    DISABLE_HEATER = 0x08

    # Gas measurement settings
    DISABLE_GAS_MEAS = 0x00
    ENABLE_GAS_MEAS = 0x01
    
    # Over-sampling settings
    OS_NONE = 0
    OS_1X = 1
    OS_2X = 2
    OS_4X = 3
    OS_8X = 4
    OS_16X = 5

    # IIR filter settings
    FILTER_SIZE_0 = 0
    FILTER_SIZE_1 = 1
    FILTER_SIZE_3 = 2
    FILTER_SIZE_7 = 3
    FILTER_SIZE_15 = 4
    FILTER_SIZE_31 = 5
    FILTER_SIZE_63 = 6
    FILTER_SIZE_127 = 7

    # Power mode settings
    SLEEP_MODE = 0
    FORCED_MODE = 1

    # Run gas enable and disable settings
    RUN_GAS_DISABLE = 0
    RUN_GAS_ENABLE = 1
    
    # Settings selector
    OST_SEL = 1
    OSP_SEL = 2
    OSH_SEL = 4
    GAS_MEAS_SEL = 8
    FILTER_SEL = 16
    HCNTRL_SEL = 32
    RUN_GAS_SEL = 64
    NBCONV_SEL = 128
    GAS_SENSOR_SEL = GAS_MEAS_SEL | RUN_GAS_SEL | NBCONV_SEL

    # Number of conversion settings
    NBCONV_MIN = 0
    NBCONV_MAX = 9 # Was 10, but there are only 10 settings: 0 1 2 ... 8 9

    def __init__(self, addr=0x77, debug=False):
        BME680.__init__(self, addr, i2c_device=SMBus(8))
        
        #initialize for gas
        self.setGasHeaterTemperature(320)
        self.setGasHeaterDuration(150)
        self.selectGasHeaterProfile(0)
        
        self.temperature = 0
        self.pressure = 0
        self.humidity = 0
        self.gas_resistance = 0

        self._last_reading = 0
        self._min_refresh_time = 0.1

    def init(self):
        self.softReset()
        self.setPowerMode(self.SLEEP_MODE)

        self._getCalibrationData()
        
        self.setHumidityOversample(self.OS_2X)
        self.setPressureOversample(self.OS_4X)
        self.setTemperatureOversample(self.OS_8X)
        self.setFilter(self.FILTER_SIZE_3)
        self.setGasStatus(self.ENABLE_GAS_MEAS)
        self.setTempOffset(0)

    def _getCalibrationData(self):
        self._get_calibration_data()

    def softReset(self):
        self.soft_reset()

    def setTempOffset(self, value):
        self.set_temp_offset(value)

    def setHumidityOversample(self, value):
        self.set_humidity_oversample(value)

    def getHumidityOversample(self):
        return self.get_humidity_oversample()

    def setPressureOversample(self, value):
        self.set_pressure_oversample(value)

    def getPressureOversample(self):
        return self.get_pressure_oversample()

    def setTemperatureOversample(self, value):
        self.set_temperature_oversample(value)

    def getTemperatureOversample(self):
        return self.get_temperature_oversample()

    def setFilter(self, value):
        self.set_filter(value)

    def getFilter(self):
        return self.get_filter()

    def selectGasHeaterProfile(self, value):
        self.gas_heater_index = value
        self.select_gas_heater_profile(value)

    def getGasHeaterProfile(self):
        return self.get_gas_heater_profile()

    def setGasStatus(self, value):
        self.set_gas_status(value)

    def getGasStatus(self):
        return self.get_gas_status()

    def setGasHeaterProfile(self, temperature, duration, nb_profile=0):
        self.set_gas_heater_profile(temperature, duration, nb_profile)

    def setGasHeaterTemperature(self, value, nb_profile=0):
        self.gas_heater_temperature = value
        self.set_gas_heater_temperature(value, nb_profile)

    def setGasHeaterDuration(self, value, nb_profile=0):
        self.gas_heater_duration = value
        self.set_gas_heater_duration(value, nb_profile)

    def setPowerMode(self, value, blocking=True):
        self.set_power_mode(value, blocking)

    def getPowerMode(self):
        return self.get_power_mode()

    def getSensorData(self):
        if self.get_sensor_data():
            d = self.data
            
            self.temperature = d.temperature
            self.pressure = d.pressure
            self.humidity = d.humidity
            self.gas_resistance = d.gas_resistance

            self._last_reading = time.monotonic()

            return[self.temperature, self.pressure, self.humidity, self.gas_resistance]
        else:
            return [0,0,0,0]
            

    def isTime(self):
        if time.monotonic() - self._last_reading < self._min_refresh_time:
            return False
        else:
            return True

    def getTemperature(self):
        if self.isTime():
            self.getSensorData()
        return self.temperature

    def getPressure(self):
        if self.isTime():
            self.getSensorData()
        return self.pressure
        
    def getHumidity(self):
        if self.isTime():
            self.getSensorData()
        return self.humidity
        
    def getGas(self):
        if self.isTime():
            self.getSensorData()
        return self.gas_resistance

def Bme680_unittest():
    b = Bme680(0x77)
    while True:
        sensor = b.getSensorData()
        temp = b.getTemperature()
        humi = b.getHumidity()
        pressure = b.getPressure()
        gas = b.getGas()
        print("sensor data = %d, %d, %d, %d"%(sensor[0],sensor[1],sensor[2],sensor[3]))
        print("temp : %d [C], humi : %d [%%] , pressure : %d [hpa], gas : %d [ohm]"%(temp,humi,pressure,gas))
        time.sleep(0.5)

##################################################################
#Flame Sensor Class
class Flame(Input):
    def __init__(self, n):
        super().__init__(n, False)

def flame_unittest1():
    flame = Flame(2)

    while (True):
        print("%d"%(flame.read()))
        delay(50)

def flame_unittest2():
    flame = Flame(2)
    count = 0

    while (True):
        if (flame.read()):
            count += 1
            print("flame = %d"%(count))
            delay(1000)
        else:
            delay(50)

def onFlame(unuse):
    onFlame.count += 1
    print("flame = %d"%(onFlame.count))
    delay(1000)

onFlame.count = 0

def flame_unittest3():
    flame = Flame(2)
    flame.setCallback(onPir, None, Flame.RISING)
    input("Press <ENTER> key...\n")

##################################################################
# Light IC 
class Light(I2c):
    BH1750_address = 0x23

    def __init__(self, addr=BH1750_address):
        super().__init__(addr)
        self.init()

    def __del(self):
        super().__del__()

    def init(self):
        I2c.write(self,0x10)

    def getLight(self):
        data = [0,0]
        data[0] = I2c.read(self)
        data[1] = I2c.read(self)
        return ((data[0] << 8) | data[1]) / 1.2

def Light_unittest():
    l = Light(0x23)

    while True:
        print("light : %d [lx]"%l.getLight())
        time.sleep(0.5)   


##################################################################
class CO2(SpiAdc):
    def __init__(self, channel, device=0, bus=0, speed=500000):
        super().__init__(channel, device, bus, speed)

        self.setSample(1024)

    def calcPPM(self):

        return self.readVolt()/0.0004

def CO2_unittest():

    c = CO2(2)

    while True:
        print(c.read(), c.readVolt(), c.calcPPM())
        time.sleep(0.1)   

##################################################################

class Thermopile(PopThread):

    OBJECT	= 0xA0		# COMMAND(Read Object Temp.)
    SENSOR	= 0xA1		# COMMAND(Read Sensor Temp.)
    LASER	= 0xA4		# COMMAND(Laser ON)

    def __init__(self, n = 7, device=0, bus=0, speed=1000000):
        self._sec = n #SCE Pin
        self._device = device
        self._bus = bus
        self._spi = spidev.SpiDev()
        self._spi.open(self._bus,self._device)
        self._spi.mode = 3
        self._spi.max_speed_hz = speed
        self._delay = 15
        GPIO.setup(self._sec, GPIO.OUT)
        self.SCE_HIGH()

        time.sleep(0.5)
        self.mode = True
        self.iSensor = 0
        self.iObject = 0
        self.count=0
        self.laserOn()
        self.start()

    def SCE_HIGH(self):
        GPIO.output(self._sec, GPIO.HIGH)

    def SCE_LOW(self):
        GPIO.output(self._sec, GPIO.LOW)

    def __del(self):
        if self.mode == True:
            self().stop()

        self._spi.close()

    def sendCommand(self, ADR):

        Data_Buf = [[ADR],[0x22],[0x22]]
        
        self.SCE_LOW()  				# SCE LOW
        time.sleep(0.00001)				# delay 10us

        self._spi.xfer2(Data_Buf[0])
        time.sleep(0.00001)				# delay 10us
        
        data = self._spi.xfer2(Data_Buf[1])
        Data_Buf[1] = data[-1]
        time.sleep(0.00001)				# delay 10us
        data = self._spi.xfer2(Data_Buf[2])
        Data_Buf[2] = data[-1]
        time.sleep(0.00001)				# delay 10us

        self.SCE_HIGH()  				# SCE HIGH
        return (Data_Buf[2]*256+Data_Buf[1])			# High + Lo byte

    def read(self):
        self.iSensor = self.sendCommand(self.SENSOR)
        time.sleep(0.02)
        self.iObject = self.sendCommand(self.OBJECT)
        time.sleep(0.05)
        return self.iObject/10

    def readSensor(self):
        self.read()
        return self.iSensor/10

    def laserOn(self):
        self.sendCommand(self.LASER)

    def setInterval(self, n):
        self._delay=n

    def run(self):
        if self.count == self._delay:
            self.laserOn();
            self.count=0;
        else:
            self.count = self.count+1
            time.sleep(1)

    def setChipSelect(self, cs):
        self._sec = cs
        GPIO.setup(self._sec, GPIO.OUT)
        self.SCE_HIGH()

        self.laserOn()	

        if self.mode == True:
            self.start()
        else:
            self.stop()

    def setLaserAutomode(self, Automode):
        self.mode = Automode

        if self.mode==True:
            self.start()
        elif self.mode==False:
            self.stop()

def Thermopile_unittest1():
    d = Thermopile(7)

    while True :
        print(d.read())
        time.sleep(1)    

def Thermopile_unittest2():
    d = Thermopile(7)
    count = 0

    d.setLaserAutomode(False)

    while True :

        count = count+1

        print(count)

        if count>30:
            count = 0
            d.laserOn()

        time.sleep(1)   

class MicroWave:
    AVERAGE = 2
    DOPPLER_DIV = 19
    samples = [0,0]

    def __init__(self, n):
        self._gpio = n
        GPIO.setup(self._gpio, GPIO.IN)
        time.sleep(0.1)

    def read(self):
        while GPIO.input(self._gpio)==False:
            pass

        for x in range(self.AVERAGE):
            start = time.time()

            while GPIO.input(self._gpio)==False:
                pass
            while GPIO.input(self._gpio)==True:
                pass

            end = time.time()

            duration = end-start

            self.samples[x] = duration*1000000

        # Check for consistency
        samples_ok = True

        nbPulsesTime = self.samples[0]

        for x  in range(1,self.AVERAGE,1):
            nbPulsesTime = nbPulsesTime+self.samples[x]
            if ((self.samples[x] > self.samples[0] * 2) or (self.samples[x] < self.samples[0] / 2)):
                samples_ok = False

        if (samples_ok):
            Ttime = nbPulsesTime / self.AVERAGE
            Freq = 1000000 / Ttime

            hz = Freq
            speed = Freq/self.DOPPLER_DIV

            time.sleep(0.1)

            return speed
        else:
            hz = 0
            speed =0
            return 0

def MicroWave_unittest():
    m = MicroWave(25)

    while True:
        result = m.read()

        if result!=0:
            print("%5.5f [km/h]\r\n"%(result))			

        else:
            print(".\n")
        
        time.sleep(0.5)




##################################################################
# SoundMeter classes

class SoundMeter:
    def __init__(self, sampleFormat=pyaudio.paInt16, channelNums=1, framesPerBuffer=1024, sampleRate=48000):
        self.func = None
        self.args = None
        self.isStop = False

        self.p = pyaudio.PyAudio()

        self.stream = self.p.open(format=sampleFormat, channels=channelNums, rate=sampleRate, input=True, 
                                  frames_per_buffer=framesPerBuffer, stream_callback=self._callback)
    
    def __del__(self):
        self.stop()

    def setCallback(self, func, *args):
        self.func = func
        self.args = args
        self.stream.start_stream()

    def stop(self):
        self.isStop = True
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    def _callback(self, inData, frameCount, timeInfo, status):
        rms = audioop.rms(inData, 2)
        self.func(rms, inData, *self.args)

        data = chr(0) * len(inData)
        return (data, pyaudio.paContinue if not self.isStop else pyaudio.paAbort)

##################################################################
# IRRemote classes

class IRRemote:    
    def __init__(self, pin=16, callback = None):        
        self.decoding = False
        self.pList = []
        self.timer = time.time()
        if callback == 'DECODE':
            self.callback = self.print_ir_code
        else:
            self.callback = callback
        self.checkTime = 150  # time in milliseconds
        self.verbose = False
        self.repeatCodeOn = True
        self.lastIRCode = 0
        self.maxPulseListLength = 70
        self.gpio=pin
        GPIO.setup(self.gpio,GPIO.IN)
        GPIO.add_event_detect(self.gpio,GPIO.BOTH,callback=self.pWidth)

    def __del__(self):
        self.remove_callback()
        GPIO.cleanup(self.gpio)

    def cleanup_pin(self):
        GPIO.cleanup(self.gpio)

    def pWidth(self, pin):
        self.pList.append(time.time()-self.timer)
        self.timer = time.time()        

        if self.decoding == False:
            self.decoding = True
            check_loop = Thread(name='self.pulse_checker',target=self.pulse_checker)
            check_loop.start()           
            
        return

    def pulse_checker(self):
        timer = time.time()

        while True:                
                check = (time.time()-timer)*1000
                if check > self.checkTime:                    
                    #print(check, len(self.pList))
                    break
                if len(self.pList) > self.maxPulseListLength:
                    #print(check, len(self.pList))
                    break
                time.sleep(0.001)

        if len(self.pList) > self.maxPulseListLength:
            decode = self.decode_pulse(self.pList)
            self.lastIRCode = decode

        # if the length of self.pList is less than 10
        # assume repeat code found
        elif len(self.pList) < 10:
            if self.repeatCodeOn == True:
                decode = self.lastIRCode
            else:
                decode = 0
                self.lastIRCode = decode
        else:
            decode = 0
            self.lastIRCode = decode

        self.pList = []
        self.decoding = False

        if self.callback != None:
            self.callback(decode)
        
        return

    def decode_pulse(self,pList):
        bitList = []
        sIndex = -1
        for p in range(0,len(pList)):
            try:
                pList[p]=float(pList[p])*1000
                if self.verbose == True:
                    print(pList[p])
                if pList[p]<11:
                    if sIndex == -1:
                        sIndex = p
            except:            
                pass

        if sIndex == -1:
            return -1

        if sIndex+1 >= len(pList):
            return -1
        
        if (pList[sIndex]<4 or pList[sIndex]>11):
            return -1

        if (pList[sIndex+1]<2 or pList[sIndex+1]>6):
            return -1

        for i in range(sIndex+2,len(pList),2):
            if i+1 < len(pList):
                if pList[i+1]< 0.9:  
                    bitList.append(0)
                elif pList[i+1]< 2.5:
                    bitList.append(1)
                elif (pList[i+1]> 2.5 and pList[i+1]< 45):
                    break
                else:
                    break

        if self.verbose == True:
            print(bitList)

        pulse = 0
        bitShift = 0

        for b in bitList:            
            pulse = (pulse<<bitShift) + b
            bitShift = 1        

        return pulse

    def set_callback(self, callback = None):
        self.callback = callback
        return

    def remove_callback(self):
        self.callback = None
        return

    def print_ir_code(self, code):
        print(hex(code))
        return

    def set_verbose(self, verbose = True):
        self.verbose = verbose
        return

    def set_repeat(self, repeat = True):
        self.repeatCodeOn = repeat
        return



from random import *

if __name__ == "__main__":
    pass
    #popmultitask_unittest()
    #led_unittest()
    #laser_unittest()
    #buzzer_unittest()
    #relay_unittest1()
    #relay_unittest2()
    #ledex_unittest1()
    #ledex_unittest2()
    #rgbled_unittest1()
    #rgbled_unittest2()
    #rgbled_unittest3()
    #dcmotor_unittest1()
    #dcmotor_unittest2()
    #stepmotor_unittest1()
    #stepmotor_unittest2()
    #stepmotor_unittest3()
    #switch_unittest1()
    #switch_unittest2()
    #switch_unittest3()
    #touch_unittest1()
    #touch_unittest2()
    #touch_unittest3()
    #reed_unittest1()
    #reed_unittest2()
    #reed_unittest3()
    #limitswitch_unittest1()
    #limitswitch_unittest2()
    #limitswitch_unittest1()
    #limitswitch_unittest2()
    #limitswitch_unittest3()
    #mercury_unittest1()
    #mercury_unittest2()
    #mercury_unittest3()
    #knock_unittest1()
    #knock_unittest2()
    #knock_unittest3()
    #tilt_unittest1()
    #tilt_unittest2()
    #tilt_unittest3()
    #shock_unittest1()
    #shock_unittest2()
    #shock_unittest3()
    #opto_unittest1()
    #opto_unittest2()
    #opto_unittest3()
    #pir_unittest1()
    #pir_unittest2()
    #pir_unittest3()
    #flame_unittest1()
    #flame_unittest2()
    #flame_unittest3()
    #linetrace_unittest1()
    #linetrace_unittest2()
    #linetrace_unittest3()
    #temphumi_unittest1()
    #temphumi_unittest2()
    #ultrasonic_unittest1()
    #ultrasonic_unittest2()
    #ultrasonic_unittest3()
    #spiadc_unittest1()
    #spiadc_unittest2()
    #spiadc_unittest3()
    #spiadc_unittest4()
    #spiadc_unittest5()
    #shock2_unittest1()
    #shock2_unittest2()
    #shock2_unittest3()
    #sound_unittest1()
    #sound_unittest2()
    #sound_unittest3()
    #potentiometer_unittest1()
    #potentiometer_unittest2()
    #potentiometer_unittest3()
    #potentiometer_unittest4()
    #potentiometer_unittest5()
    #cds_unittest1()
    #cds_unittest2()
    #cds_unittest3()
    #cds_unittest4()
    #soilmoisture_unittest1()
    #soilmoisture_unittest2()
    #soilmoisture_unittest3()
    #thermistor_unittest1()
    #thermistor_unittest2()
    #thermistor_unittest3()
    #thermistor_unittest4()
    #temperature_unittest1()
    #temperature_unittest2()
    #temperature_unittest3()
    #gas_unittest1()
    #gas_unittest2()
    #gas_unittest3()
    #gas_unittest4()
    #dust_unittest1()
    #dust_unittest2()
    #dust_unittest3()
    #psd_unittest1()
    #psd_unittest2()
    #psd_unittest3()
    #psd_unittest4()
    #shiftRegister_unittest()
    #i2c_unittest()
    #At42qt1070_unittest()
    #sht20_unittest()
    #pca9685_unittest()
    #mpu6050_unittest()
    #fan_unittest()
    #textlcd_unittest()
    #ledstrip_unittest()
