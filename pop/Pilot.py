
__version__='1.2.1'

''' Pop AutoCar '''
import time
import math
import smbus2 as smbus
import random
import ipywidgets.widgets as widgets
import __main__
import subprocess as sp

if not "pwm_time_log" in dir(__main__):
    __main__.pwm_time_log = 0

class PWM(object):
    _mode_adr              = 0x00  # Mode 1 Reg
    _base_adr_low          = 0x08 
    _base_adr_high         = 0x09
    _prescale_adr          = 0xFE  # Prescale Reg

    def __init__(self, bus, address, wait_i2c=True):
        '''
        I2C 버스에 연결된 PWM 컨트롤러 인스턴스 생성
        @param bus: PWM 컨트롤러가 연결된 I2C 버스 번호(0, 1).
        @param address: PWM 컨트롤러 주소(기본값은 0x41)
        '''
        self.wait_i2c=wait_i2c
        self.bus = smbus.SMBus(bus)
        self.address = address
        self._writeByte(self._mode_adr, 0x00)

    def setFreq(self, freq):
        '''
        PWM 주파수 설정
        @param freq: Hz 단위 주파수
        '''
        prescaleValue = 25000000.0    # 25MHz
        prescaleValue /= 4096.0       # 12-bit
        prescaleValue /= float(freq)
        #prescaleValue -= 1.0
        prescale = math.floor(prescaleValue + 0.5)
        if prescale < 3:
            raise ValueError("주파수 설정 오류")

        oldmode = self._readByte(self._mode_adr)
        newmode = (oldmode & 0x7F) | 0x10 # mode 1, sleep
        self._writeByte(self._mode_adr, newmode)
        self._writeByte(self._prescale_adr, int(math.floor(prescale)))
        self._writeByte(self._mode_adr, oldmode)
        time.sleep(0.005)
        self._writeByte(self._mode_adr, oldmode | 0xA1) #mode 1, autoincrement on (old 0x80)

    def setDuty(self, channel, duty):
        '''
        PWM 채널의 듀티비 설정
        @param channel: 채널 번호 (0~15)
        @param duty: 튜티비 (0~100)
        '''
        data = int(duty * 4096 / 100) # 0..4096 (included)
        
        if self.wait_i2c:
            while time.time()-__main__.pwm_time_log<0.05: time.sleep(0.01)
        self._writeByte(self._base_adr_low + 4 * channel, data & 0xFF)
        self._writeByte(self._base_adr_high + 4 * channel, data >> 8)
        __main__.pwm_time_log=time.time()

    def _writeByte(self, reg, value):
        try:
            self.bus.write_byte_data(self.address, reg, value)
        except Exception as e:
            v=ValueError("[Errno "+str(e.errno)+"] An error occured while reading I2C Devcie")
            v.errno=e.errno
            raise v

    def _readByte(self, reg):
        try:
            result = self.bus.read_byte_data(self.address, reg)
            return result
        except Exception as e:
            v=ValueError("[Errno "+str(e.errno)+"] An error occured while reading I2C Devcie")
            v.errno=e.errno
            raise v


'''
0 : AutoCar
1 : AutoCar Racing
2 : SerBot
3 : AutoCar Prime
4 : SerBot Prime X
5 : AutoCar Prime X
6 : AutoCar Prime + NX
'''
_cat = 0

try :
    _p=PWM(0,0x5d) #AutoCar
    __main__._camera_flip_method='0'
except Exception as e:
    if e.errno == 121:
        try :
            _p=PWM(0,0x40) #SerBot
            __main__._camera_flip_method='2'
            _cat = 2
        except Exception as e:
            if e.errno == 121:
                try :
                    _p=PWM(0,0x5c) #AutoCar Prime
                    __main__._camera_flip_method='0'
                    _cat = 3
                except:
                    try:
                        _p=PWM(1,0x5c) #AutoCar Prime + NX
                        __main__._camera_flip_method='0'
                        _cat = 6
                    except Exception as e:
                        if e.errno == 121:
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
                                if e.errno==19:
                                    _cat = 1 #AutoCar Racing
                                    __main__._camera_flip_method='0'

_p=None

has_lidar="10c4:ea60" in str(sp.check_output(["lsusb"], shell=True))


class Driving:
    '''
    t       duty cycle          speed
    0.72ms   0.72ms/20ms = 3.6%   0 degs
    1.36ms   1.36ms/20ms = 6.8%   0
    2.0ms   2.0ms/20ms = 10%  100%
    '''

    GPIO_ESC = 12

    GPIO_LEFT_FORWARD    = 0
    GPIO_LEFT_BACKWARD   = 1
    GPIO_RIGHT_FORWARD   = 2
    GPIO_RIGHT_BACKWARD  = 3

    MIN_SPEED = 20
    MAX_SPEED = 99

    STD_DUTY=7.4 #6.8
    DUTY_RANGE=1.2 #If you want to set the maximum performance speed of the motor, replace it to 3.2.

    STAT_STOP = 1
    STAT_FORWARD = 2
    STAT_BACKWARD = 3

    stat=1

    def __init__(self, bus, addr, freq):
        if _cat==0 or _cat==1 or _cat==3 or _cat==6:
            self.stat = Driving.STAT_STOP
            self.pwm = PWM(bus, addr)
            self.pwm.setFreq(freq)
            self.speed = Driving.MIN_SPEED
        elif _cat==5:
            global CAN
            from pop import CAN
            self._can=CAN.Car()

    def __del__(self):
        if _cat==1:
            self.pwm.setDuty(Driving.GPIO_ESC, Driving.STD_DUTY)
        elif _cat==0 or _cat==3 or _cat==6:
            self.pwm.setDuty(Driving.GPIO_RIGHT_FORWARD, 0)
            self.pwm.setDuty(Driving.GPIO_LEFT_FORWARD, 0)
            self.pwm.setDuty(Driving.GPIO_RIGHT_BACKWARD, 0)
            self.pwm.setDuty(Driving.GPIO_LEFT_BACKWARD, 0)
        elif _cat==5:
            self._can.wheel(0)
            self._can.camPan(0)
            self._can.camTilt(0)
            self._can.steer(0)

    def setSpeed(self, speed):
        if self.stat != Driving.STAT_STOP:
            self.direction(self.stat, speed)

    def getSpeed(self):
        return self.speed

    def stop(self):
        if _cat==1:
            self.pwm.setDuty(Driving.GPIO_ESC, Driving.STD_DUTY)
        elif _cat==0 or _cat==3 or _cat==6:
            self.pwm.setDuty(Driving.GPIO_RIGHT_BACKWARD, 0)
            self.pwm.setDuty(Driving.GPIO_LEFT_FORWARD, 0)
            self.pwm.setDuty(Driving.GPIO_RIGHT_FORWARD, 0)
            self.pwm.setDuty(Driving.GPIO_LEFT_BACKWARD, 0)
        elif _cat==5:
            self._can.wheel(0)

        self.stat = Driving.STAT_STOP

    def direction(self, stat, speed):
        if speed:
            if speed > Driving.MAX_SPEED:
                speed = Driving.MAX_SPEED
            elif speed < Driving.MIN_SPEED:
                speed = Driving.MIN_SPEED
            self.speed = speed

        if _cat==1:
            speed_rate = speed/Driving.MAX_SPEED
            speed_duty = (speed_rate * Driving.DUTY_RANGE)

            if stat == Driving.STAT_BACKWARD:
                duty = Driving.STD_DUTY - speed_duty
                if self.stat != Driving.STAT_BACKWARD:
                    self.pwm.setDuty(Driving.GPIO_ESC, duty)
                    self.pwm.setDuty(Driving.GPIO_ESC, Driving.STD_DUTY)
                    self.pwm.setDuty(Driving.GPIO_ESC, duty)
                else:
                    self.pwm.setDuty(Driving.GPIO_ESC, duty)

                self.stat = Driving.STAT_BACKWARD
            else:
                duty = Driving.STD_DUTY + speed_duty
                self.stat = Driving.STAT_FORWARD
                self.pwm.setDuty(Driving.GPIO_ESC, duty)
        elif _cat==0 or _cat==3 or _cat==6:
            if stat == Driving.STAT_BACKWARD:
                self.stat = Driving.STAT_BACKWARD
                self.pwm.setDuty(Driving.GPIO_RIGHT_BACKWARD, 0)
                self.pwm.setDuty(Driving.GPIO_LEFT_FORWARD, 0)
                self.pwm.setDuty(Driving.GPIO_RIGHT_FORWARD, self.speed)
                self.pwm.setDuty(Driving.GPIO_LEFT_BACKWARD, self.speed)
            else:
                self.stat = Driving.STAT_FORWARD
                self.pwm.setDuty(Driving.GPIO_RIGHT_FORWARD, 0)
                self.pwm.setDuty(Driving.GPIO_LEFT_BACKWARD, 0)
                self.pwm.setDuty(Driving.GPIO_RIGHT_BACKWARD, self.speed)
                self.pwm.setDuty(Driving.GPIO_LEFT_FORWARD, self.speed)
        elif _cat==5:
            if stat == Driving.STAT_BACKWARD:
                self.stat = Driving.STAT_BACKWARD
                self._can.wheel(-self.speed)
            else:
                self.stat = Driving.STAT_FORWARD
                self._can.wheel(self.speed)

class Wheel:
    GPIO_SERVO  = 15

    '''
    t       duty cycle          direction
    0.5ms   0.5ms/20ms = 2.5%   0 degs
    1.5ms   1.5ms/20ms = 7.5%   90 degs
    2.5ms   2.5ms/20ms = 12.5%  180 degs
    '''
    MIN_VECTOR  = 2.5
    WITH_VECTOR = 10    # 12.5% - 2.5%

    def __init__(self, bus, addr, freq):
        if _cat==0 or _cat==1 or _cat==3 or _cat==6:
            self.pwm = PWM(bus, addr)
            self.pwm.setFreq(freq)
        elif _cat==5:
            global CAN
            from pop import CAN
            self._can=CAN.Car()

        self.centerAngle = 90

    def _angle2duty(self, n):
        return (Wheel.WITH_VECTOR / 180) * n + Wheel.MIN_VECTOR

    def setCenterAngle(self, angle):
        self.centerAngle = angle
    
    def turnLeft(self, angle):
        a = self.centerAngle - angle

        if _cat==1:
            a = self.centerAngle + angle
            
        if _cat==0 or _cat==1 or _cat==3 or _cat==6:
            self.pwm.setDuty(Wheel.GPIO_SERVO, self._angle2duty(a))
            time.sleep(angle / 100)
        elif _cat==5:
            self._can.steer((a-90)/self._can.steer_range)

    def turnRight(self, angle):
        a = self.centerAngle + angle
        
        if _cat==1:
            a = self.centerAngle - angle

        if _cat==0 or _cat==1 or _cat==3 or _cat==6:
            self.pwm.setDuty(Wheel.GPIO_SERVO, self._angle2duty(a))
            time.sleep(angle / 100)
        elif _cat==5:
            self._can.steer((a-90)/self._can.steer_range)
          
    def turnCenter(self):
        if _cat==0 or _cat==1 or _cat==3 or _cat==6:
            self.pwm.setDuty(Wheel.GPIO_SERVO, self._angle2duty(self.centerAngle)) 
        elif _cat==5:
            self._can.steer((self.centerAngle-90)/self._can.steer_range)

class CameraPod:
    Vertical_GPIO  = 14
    Horizontal_GPIO = 13

    def __init__(self, bus=0, addr=0x5d, freq=50):
        if _cat==0 or _cat==1 or _cat==3 or _cat==6:
            self.pwm = PWM(bus, addr)
            self.pwm.setFreq(freq)
            if _cat==3 or _cat==6:
                self.panCenterAngle = 188
            else:
                self.panCenterAngle = 90
            self.tiltCenterAngle = 13
        elif _cat==5:
            global CAN
            from pop import CAN
            self._can=CAN.Car()
            self.panCenterAngle = 0
            self.tiltCenterAngle = 0

    def _angle2duty(self, n):
        return n/17.3239 + 1.97

    def panLeft(self, value):
        n=self.panCenterAngle+value

        if _cat==0 or _cat==1 or _cat==3 or _cat==6:
            self.pwm.setDuty(self.Vertical_GPIO,self._angle2duty(n))
        elif _cat==5:
            self._can.camPan(n)

    def panRight(self, value):
        n=self.panCenterAngle-value

        if _cat==0 or _cat==1 or _cat==3 or _cat==6:
            self.pwm.setDuty(self.Vertical_GPIO,self._angle2duty(n))
        elif _cat==5:
            self._can.camPan(n)

    def pan(self, n):
        if _cat==0 or _cat==1 or _cat==3 or _cat==6:
            self.pwm.setDuty(self.Vertical_GPIO,self._angle2duty(self.panCenterAngle-n))
        elif _cat==5:
            self._can.camPan(n+self.panCenterAngle-90)

    def tiltBack(self):
        if _cat==0 or _cat==1 or _cat==3 or _cat==6:
            self.pwm.setDuty(self.Horizontal_GPIO,self._angle2duty(90+(not has_lidar * 90)+self.tiltCenterAngle))
        elif _cat==5:
            self._can.camTilt(90)

    def tiltFront(self):
        if _cat==0 or _cat==1 or _cat==3 or _cat==6:
            self.pwm.setDuty(self.Horizontal_GPIO,self._angle2duty(self.tiltCenterAngle))
        elif _cat==5:
            self._can.camTilt(0)

    def tilt(self, n):
        if _cat==0 or _cat==1 or _cat==3 or _cat==6:
            self.pwm.setDuty(self.Horizontal_GPIO,self._angle2duty(n+self.tiltCenterAngle))
        elif _cat==5:
            self._can.camTilt(n+self.tiltCenterAngle)

class axis6:
    address=0x68

    PW_MGMT_1 = 0x6b
    PW_MGMT_2 = 0x6c

    bus = None

    def __init__(self, bus=1):
        if _cat==0 or _cat==1 or _cat==3:
            self.bus = smbus.SMBus(bus)
            self.bus.write_byte_data(self.address, self.PW_MGMT_1, 0)
        if _cat==6:
            self.bus = smbus.SMBus(8)
            self.bus.write_byte_data(self.address, self.PW_MGMT_1, 0)
        elif _cat==4 or _cat==5 :
            self.bus = smbus.SMBus(8)
            self.bus.write_byte_data(self.address, self.PW_MGMT_1, 0)
        else:
            del self

    def __del__(self):
        self.bus.close()

    def read_word(self, adr):
        while time.time()-__main__.pwm_time_log<0.05: time.sleep(0.01)
        high = self.bus.read_byte_data(self.address, adr)
        low = self.bus.read_byte_data(self.address, adr+1)
        val = (high << 8) + low
        __main__.pwm_time_log=time.time()
        return val

    def read_word_2c(self, adr):
        #for _ in range(10):
            #try:
        val = self.read_word(adr)
        if (val >= 0x8000):
            return -((65535 - val) + 1)
        else:
            return val 
            #except Exception as e:
                #if e.errno == 110:
                #    print("I2C connection timed out. Try again after 0.5sec...\nReset the i2c bus if it continues to occur.")
                #    time.sleep(0.5)
                #pass

    def getGyro(self, axis=None):
        if type(axis)==str:
            if axis.lower()=="x":
                return self.read_word_2c(0x43)
            elif axis.lower()=="y":
                return self.read_word_2c(0x45)
            elif axis.lower()=="z":
                return self.read_word_2c(0x47)
            else:
                x=self.read_word_2c(0x43)
                y=self.read_word_2c(0x45)
                z=self.read_word_2c(0x47)
                return {"x":x, "y":y, "z":z}
        else:
            x=self.read_word_2c(0x43)
            y=self.read_word_2c(0x45)
            z=self.read_word_2c(0x47)
            return {"x":x, "y":y, "z":z}

    def getAccel(self, axis=None):
        if type(axis)==str:
            if axis.lower()=="x":
                return self.read_word_2c(0x3b)
            elif axis.lower()=="y":
                return self.read_word_2c(0x3d)
            elif axis.lower()=="z":
                return self.read_word_2c(0x3f)
            else:
                x=self.read_word_2c(0x3b)
                y=self.read_word_2c(0x3d)
                z=self.read_word_2c(0x3f)
                return {"x":x, "y":y, "z":z}
        else:
            x=self.read_word_2c(0x3b)
            y=self.read_word_2c(0x3d)
            z=self.read_word_2c(0x3f)
            return {"x":x, "y":y, "z":z}

class AutoCar(axis6):
    joystick = None
    steer_limit = 30
    max_speed=99
    min_speed=20

    def __init__(self, bus=0):
        super().__init__()

        if _cat==3 or _cat==6:
            if _cat==6: bus=1
            self.wheel = Wheel(bus, 0x5c, 50)
            self.campod = CameraPod(bus, 0x5c, 50)
        else:
            self.wheel = Wheel(bus, 0x5d, 50)
            self.campod = CameraPod(bus, 0x5d, 50)

        if _cat==1:
            self.drv = Driving(bus, 0x5d, 50)
        elif _cat==0 or _cat==3:
            self.drv = Driving(bus, 0x5e, 200)
        elif _cat==6:
            self.drv = Driving(1, 0x5e, 200)
        else:
            self.drv = Driving(None, None, None)

        if _cat==5:
            from pop import CAN
            self._can=CAN.Car()

            def us():
                return self._can.read()

            self.getUltrasonic=us
        

        self._steering = 0.0

        self.joystick=joystick(callback=self._control_as_joystick)

    @property
    def steering(self):
        return self._steering

    @steering.setter
    def steering(self, value):
        if value > 1.0 or value < -1.0:
            print("Warning : This value is out of range -1.0 to 1.0. It was adjusted to maximum.")

            if value > 1.0:
                value = 1.0
            elif value < -1.0:
                value = -1.0

        self._steering = value

        if _cat==3 or _cat==6:
            value*=-1

        if _cat==0 or _cat==1 or _cat==3 or _cat==6:
            if value < 0:
                self.wheel.turnLeft(int(abs(value) * self.steer_limit))
            else:
                self.wheel.turnRight(int(value * self.steer_limit))
        elif _cat==5:
            self.wheel._can.steer(value)
    
    @property
    def steering_gain(self):
        return self._steering_gain

    @steering_gain.setter
    def steering_gain(self, value):
        self._steering_gain = value

    def setCenterAngle(self, angle):
        self.wheel.setCenterAngle(angle)

    def correctError(self, value):
        self.setCenterAngle(self.wheel.centerAngle + value * self.steer_limit)

    def turnLeft(self):
        self.steering = -1

    def turnRight(self):
        self.steering = 1

    def turnCenter(self):
        self.steering = 0

    def setSpeed(self, speed):
        self.drv.setSpeed(speed)

    def getSpeed(self):
        return self.drv.getSpeed()

    def stop(self):
        self.drv.stop()
    
    def forward(self, speed=None):
        self.drv.direction(Driving.STAT_FORWARD, speed)
    
    def backward(self, speed=None):
        self.drv.direction(Driving.STAT_BACKWARD, speed)

    def camPan(self, value):
        self.campod.pan(value)

    def camTilt(self, value):
        if has_lidar and value>90: value=90
        self.campod.tilt(value)

    def cam2Back(self):
        self.campod.tiltBack()

    def cam2Front(self):
        self.campod.tiltFront()

    def cam2Left(self, value):
        self.campod.panLeft(value)

    def cam2Right(self, value):
        self.campod.panRight(value)

    def _control_as_joystick(self, value):
        if value['sep'] == "j":
            self.steering=value['x']
            speed=value['y']*(self.max_speed-self.min_speed)

            if speed>0:
                self.forward(speed+self.min_speed)
            elif speed<0:
                self.backward(-speed+self.min_speed)
            else:
                self.stop()

    def joystick(self):
        display(self.joystick())


class Driver:

    GPIO_ESC = 12

    GPIO_WHL_1_FORWARD    = 0
    GPIO_WHL_1_BACKWARD   = 1
    GPIO_WHL_2_FORWARD    = 2
    GPIO_WHL_2_BACKWARD   = 3
    GPIO_WHL_3_FORWARD    = 4
    GPIO_WHL_3_BACKWARD   = 5

    MIN_SPEED = 20
    MAX_SPEED = 99

    STEER_LIMIT = 180

    STAT_STOP = 1
    STAT_MOVING = 2
    STAT_SETTING = 3
    STAT_DRIVING = 4

    stat=1

    def __init__(self, bus, addr, freq):
        self.stat = Driver.STAT_STOP
        self.speed = Driver.MIN_SPEED
        self.drct = 0
        self.steer = 0

        try:
            self.pwm = PWM(bus, addr, wait_i2c=False)
            self.pwm.setFreq(freq)
        except:
            pass

    def __del__(self):
        self.stop()

    def whl(self, id, value):
        if id == 1 :
            if value < 0 :
                self.pwm.setDuty(0,abs(value))
                self.pwm.setDuty(1,0)
            elif value > 0 :
                self.pwm.setDuty(0,0)
                self.pwm.setDuty(1,abs(value))
            else :
                self.pwm.setDuty(0,0)
                self.pwm.setDuty(1,0)
        elif id == 2 :
            if value < 0 :
                self.pwm.setDuty(2,abs(value))
                self.pwm.setDuty(3,0)
            elif value > 0 :
                self.pwm.setDuty(2,0)
                self.pwm.setDuty(3,abs(value))
            else :
                self.pwm.setDuty(2,0)
                self.pwm.setDuty(3,0)
        elif id == 3 :
            if value < 0 :
                self.pwm.setDuty(4,abs(value))
                self.pwm.setDuty(5,0)
            elif value > 0 :
                self.pwm.setDuty(4,0)
                self.pwm.setDuty(5,abs(value))
            else :
                self.pwm.setDuty(4,0)
                self.pwm.setDuty(5,0)

    def setSpeed(self, speed):
        if speed:
            if abs(speed) > Driver.MAX_SPEED:
                if speed > 0 :
                    speed = Driver.MAX_SPEED
                elif speed < 0 :
                    speed = -Driver.MAX_SPEED
            elif abs(speed) < Driver.MIN_SPEED:
                if speed > 0 :
                    speed = Driver.MIN_SPEED
                elif speed < 0 :
                    speed = -Driver.MIN_SPEED
            self.speed = speed

        if self.stat == Driver.STAT_MOVING:
            self.move(self.drct, self.speed)
        elif self.stat == Driver.STAT_DRIVING:
            self.drive(self.steer, self.speed)

    def getSpeed(self):
        return self.speed

    def stop(self):
        self.whl(1,0)
        self.whl(2,0)
        self.whl(3,0)

        self.stat = Driver.STAT_STOP

    def setDirection(self, degree):
        self.drct = degree % 360

        if self.stat == Driver.STAT_MOVING:
            self.move(self.drct, self.speed)

    def setSteer(self, degree):
        if abs(degree) > self.STEER_LIMIT :
            if degree > 0 :
                degree = self.STEER_LIMIT
            elif degree < 0 :
                degree = -self.STEER_LIMIT

        self.steer=degree

        if self.stat == Driver.STAT_DRIVING:
            self.drive(self.steer, self.speed)

    def move(self, degree=None, speed=None):
        self.stat = Driver.STAT_SETTING

        if degree is None :
            degree = self.drct
        else :
            self.setDirection(degree)

        if speed is None :
            speed = self.speed
        elif speed == 0 :
            self.stop()
        else :
            self.setSpeed(speed)
        
        w1=math.sin(math.radians(self.drct-300))
        w2=math.sin(math.radians(self.drct-60))
        w3=math.sin(math.radians(self.drct-180))
        
        rate = (1.0/max(abs(w1), abs(w2), abs(w3)))

        w1*=rate*self.speed
        w2*=rate*self.speed
        w3*=rate*self.speed

        self.whl(1,w1)
        self.whl(2,w2)
        self.whl(3,w3)

        self.stat = Driver.STAT_MOVING

    def drive(self, steer=None, speed=None):
        self.stat = Driver.STAT_SETTING

        if steer is None :
            steer = self.steer
        else :
            self.setSteer(steer)

        if speed is None :
            speed = self.speed
        elif speed == 0 :
            self.stop()
            return
        else :
            self.setSpeed(speed)
        
        if abs(steer) != 0 and abs(steer) != 180:
            theta = math.radians(90-steer)

            h = 500 #mm
            rad = {'x':h * math.tan(theta), 'y':0}
            m = 150 #mm

            W_1 = {'x':m * math.cos(math.radians(30)), 'y':m * math.sin(math.radians(30))}
            W_2 = {'x':m * math.cos(math.radians(150)), 'y':m * math.sin(math.radians(150))}
            W_3 = {'x':m * math.cos(math.radians(270)), 'y':m * math.sin(math.radians(270))}

            RW_1 = math.sqrt((W_1['x']-rad['x'])**2 + (W_1['y']-rad['y'])**2) * (steer/abs(steer))
            RW_2 = math.sqrt((W_2['x']-rad['x'])**2 + (W_2['y']-rad['y'])**2) * (steer/abs(steer))
            RW_3 = math.sqrt((W_3['x']-rad['x'])**2 + (W_3['y']-rad['y'])**2) * (steer/abs(steer))

            alpha_1 = math.acos( W_1['y'] / RW_1 ) + math.radians(30)
            alpha_2 = math.acos( W_2['y'] / RW_2 ) + math.radians(150)
            alpha_3 = math.acos( W_3['y'] / RW_3 ) + math.radians(270)

            w1=math.sin(alpha_1)
            w2=math.sin(alpha_2)
            w3=math.sin(alpha_3)

            rate = (1.0/max(abs(w1), abs(w2), abs(w3)))

            w1*=rate*self.speed
            w2*=rate*self.speed
            w3*=rate*self.speed

            self.whl(1,w1)
            self.whl(2,w2)
            self.whl(3,w3)
        elif abs(steer) == 180 :
            self.whl(1,self.speed)
            self.whl(2,-self.speed)
            self.whl(3,0)
        elif abs(steer) == 0 :
            self.whl(1,self.speed)
            self.whl(2,-self.speed)
            self.whl(3,0)

        self.stat = Driver.STAT_DRIVING

    def turnLeft(self):
        self.whl(1,-self.speed)
        self.whl(2,-self.speed)
        self.whl(3,-self.speed)

    def turnRight(self):
        self.whl(1,self.speed)
        self.whl(2,self.speed)
        self.whl(3,self.speed)

class SerBot(axis6):
    _joystick = None
    toggleWidget = None
    steer_limit = 90
    max_speed=99
    min_speed=20

    error = 0
    _fix=False

    def __init__(self, bus=8):
        super().__init__()

        self.drv = Driver(bus, 0x40, 200)

        if _cat==4:
            global CAN
            from pop import CAN
            self.omniwheel=CAN.OmniWheel()
            self.drv.whl=self.omniwheel.wheel

            def us():
                return self.omniwheel.read(1)

            def psd():
                return self.omniwheel.read(2)

            self.getUltrasonic=us
            self.getPSD=psd

        self.speed = self.drv.speed
        self._steering = 0.0

        self.toggleWidget=widgets.ToggleButton(
                value=False,
                description='Fixed axis',
                disabled=False,
                button_style='danger'
            )
        self.toggleWidget.observe(self._onclick_fixed_axis,'value')

        self._joystick=joystick(callback=self._control_as_joystick)

    def _onclick_fixed_axis(self, e):
        if e['new']:
            e.owner.button_style='success'
            self._fix=True
        else:
            e.owner.button_style='danger'
            self._fix=False

    @property
    def steering(self):
        return self._steering

    @steering.setter
    def steering(self, value):
        if value > 1.0 or value < -1.0:
            print("Warning : This value is out of range -1.0 to 1.0. It was adjusted to maximum.")

            if value > 1.0:
                value = 1.0
            elif value < -1.0:
                value = -1.0

        self._steering = value

        self.drv.setSteer((value + self.error) * self.steer_limit)

    def correctError(self, value):
        if value > 1.0:
            value = 1.0
        elif value < -1.0:
            value = -1.0

        self.error=value

    def turnLeft(self):
        self.drv.turnLeft()

    def turnRight(self):
        self.drv.turnRight()

    def setSpeed(self, speed):
        self.speed=speed
        self.drv.setSpeed(speed)

    def getSpeed(self):
        return self.drv.getSpeed()

    def stop(self):
        self.drv.stop()
    
    def forward(self, speed=None):
        if speed is not None:
            self.speed=speed

        self.drv.drive(self.drv.steer, self.speed)
    
    def backward(self, speed=None):
        if speed is not None:
            self.speed=speed

        self.drv.drive(self.drv.steer, -self.speed)

    def move(self, degree, speed):
        self.drv.move(degree,speed)

    def camPan(self, value):
        self.campod.pan(value)

    def camTilt(self, value):
        self.campod.tilt(value)

    def cam2Back(self):
        self.campod.tiltBack()

    def cam2Front(self):
        self.campod.tiltFront()

    def cam2Left(self, value):
        self.campod.panLeft(value)

    def cam2Right(self, value):
        self.campod.panRight(value)

    def _control_as_joystick(self, value):
        if value['sep'] == "j":
            if self._fix:
                if value['x'] == 0 and value['y'] == 0:
                    self.stop()
                else:
                    print(value['x'], ", ", value['y'])
                    theta = 90 - 90 * np.sign(value['y'])

                    if value['x'] != 0:
                        theta = (90 - math.degrees(math.atan2(value['y'],value['x']))) % 360

                    speed = math.sqrt(value['x']**2 + value['y']**2)

                    if speed > 1 :
                        speed = 1

                    speed=speed*(self.max_speed-self.min_speed)

                    self.move(theta, speed)
            else:
                self.steering=value['x']
                speed=value['y']*(self.max_speed-self.min_speed)

                if speed > 0:
                    self.forward(speed+self.min_speed)
                elif speed < 0:
                    self.backward(-speed+self.min_speed)
                else:
                    self.stop()

    def joystick(self):
        display(self.toggleWidget)
        display(self._joystick())
        

class joystick(object):
    server=None
    value={"x":0,"y":0}
    server_thread=None
    port=8885
    js=None
    id=format(int(random.uniform(0.5,1.5)*time.time()*(10**7)),'X')

    def handler(self, websocket, data):
        sep, x, y=data.split(",")
        self.value={"sep":sep, "x":float(x),"y":float(y)}
        
        if self.callback is not None:
            self.callback(self.value)
                
    def _serve(self):
        for _ in range(100):
            try:
                self.js=HTML('<style>.joystick_focused{ cursor:grabbing !important;}    .joystick_background{user-select: none;        background: #fff3f3;        border: 1px solid #ffa29e;        border-radius: 50%;        height: 12em;        width: 12em;    margin:2.5em;}    .joystick_stick{cursor:grab; user-select: none;        background: #F74138;        border-radius: 50%;        box-shadow: 0.375em 0.375em 0 0 rgba(15, 28, 63, 0.125);        height: 5em;        width: 5em;        transform: translate(50%,50%);    }</style><div><div id="joystick_background_'+self.id+'" class="joystick_background">    <div id="joystick_stick_'+self.id+'" class="joystick_stick" style="position:absolute" onmousedown="start_joystick_'+self.id+'(this); joystick_focus_'+self.id+'(this);" ondrag="joystick_'+self.id+'(e)" onmouseup="reset_joystick_'+self.id+'(this); joystick_disfocus_'+self.id+'(this);"></div></div></div><script> X=0; Y=0; function joystick_focus_'+self.id+'(e){ e.classList.add("joystick_focused");} function joystick_disfocus_'+self.id+'(e){ e.classList.remove("joystick_focused");}    var port_'+self.id+'='+str(self.port)+';    var sock_'+self.id+'=new WebSocket("ws://"+window.location.hostname+":"+port_'+self.id+');    var sw_'+self.id+'=false;    var preX_'+self.id+', preY_'+self.id+', X_'+self.id+', Y_'+self.id+', nX_'+self.id+', nY_'+self.id+';    var back_'+self.id+'=document.getElementById("joystick_background_'+self.id+'");    var stick_'+self.id+'=document.getElementById("joystick_stick_'+self.id+'");    var back_width_'+self.id+'=back_'+self.id+'.offsetWidth;    var back_height_'+self.id+'=back_'+self.id+'.offsetHeight;    var stick_width_'+self.id+'=stick_'+self.id+'.offsetWidth;    var stick_height_'+self.id+'=stick_'+self.id+'.offsetHeight;  intlog_'+self.id+'=Date.now();  setInterval(()=>{try{if (parseFloat(sX_'+self.id+')==0 && parseFloat(sY_'+self.id+')==0) {if (Date.now()-intlog_'+self.id+'<500) sock_'+self.id+'.send("j," + sX_'+self.id+' + "," + sY_'+self.id+');}else{ sock_'+self.id+'.send("j," + sX_'+self.id+' + "," + sY_'+self.id+'); intlog_'+self.id+'=Date.now();}}catch{console.log("Waiting to connect...");}},50); function move_'+self.id+'(evt){        X=evt.clientX;        Y=evt.clientY;        if(sw_'+self.id+'){            nX_'+self.id+'+=X-preX_'+self.id+';            nY_'+self.id+'+=Y-preY_'+self.id+';            preX_'+self.id+'=X;            preY_'+self.id+'=Y;            if (nX_'+self.id+'>back_width_'+self.id+'-stick_width_'+self.id+'/2) nX_'+self.id+'=back_width_'+self.id+'-stick_width_'+self.id+'/2;            else if (nX_'+self.id+'<-stick_width_'+self.id+'/2) nX_'+self.id+'=-stick_width_'+self.id+'/2;            if (nY_'+self.id+'>back_height_'+self.id+'-stick_height_'+self.id+'/2) nY_'+self.id+'=back_height_'+self.id+'-stick_height_'+self.id+'/2;            else if (nY_'+self.id+'<-stick_height_'+self.id+'/2) nY_'+self.id+'=-stick_height_'+self.id+'/2;            sX_'+self.id+'=(nX_'+self.id+'+stick_width_'+self.id+'/2-back_width_'+self.id+'/2)/(back_width_'+self.id+'/2);            sY_'+self.id+'=-(nY_'+self.id+'+stick_height_'+self.id+'/2-back_height_'+self.id+'/2)/(back_height_'+self.id+'/2);                        /*sock_'+self.id+'.send("j,"+sX_'+self.id+'+","+sY_'+self.id+');*/            stick_'+self.id+'.style.transform="translate("+nX_'+self.id+'+"px,"+nY_'+self.id+'+"px)";        }    };    function up_'+self.id+'(){joystick_disfocus_'+self.id+'(stick_'+self.id+');        if(sw_'+self.id+'){            sw_'+self.id+'=false;            reset_joystick_'+self.id+'(stick_'+self.id+');        }    };    function start_joystick_'+self.id+'(e){window.onmousemove=move_'+self.id+'; window.onmouseup=up_'+self.id+';        preX_'+self.id+'=X;        preY_'+self.id+'=Y;        sw_'+self.id+'=true;    }    function joystick_'+self.id+'(e){        nX_'+self.id+'+=X-preX_'+self.id+';        nY_'+self.id+'+=Y-preY_'+self.id+';        preX_'+self.id+'=X;        preY_'+self.id+'=Y;        if (nX_'+self.id+'>back_width_'+self.id+'-stick_width_'+self.id+'/2) nX_'+self.id+'=back_width_'+self.id+'-stick_width_'+self.id+'/2;        else if (nX_'+self.id+'<-stick_width_'+self.id+'/2) nX_'+self.id+'=-stick_width_'+self.id+'/2;        if (nY_'+self.id+'>back_height_'+self.id+'-stick_height_'+self.id+'/2) nY_'+self.id+'=back_height_'+self.id+'-stick_height_'+self.id+'/2;        else if (nY_'+self.id+'<-stick_height_'+self.id+'/2) nY_'+self.id+'=-stick_height_'+self.id+'/2;                e.style.transform="translate("+nX_'+self.id+'+"px,"+nY_'+self.id+'+"px)";    }    function reset_joystick_'+self.id+'(e){        sw_'+self.id+'=false;        nX_'+self.id+' = back_width_'+self.id+'/2 - stick_width_'+self.id+'/2;        nY_'+self.id+' = back_height_'+self.id+'/2 - stick_height_'+self.id+'/2;        e.style.transform="translate("+nX_'+self.id+'+"px,"+nY_'+self.id+'+"px)"; sX_'+self.id+'="0"; sY_'+self.id+'="0"; sock_'+self.id+'.send("j,0,0");    }    reset_joystick_'+self.id+'(stick_'+self.id+'); console.log("Loaded.");</script>')

                self.server = WebSocketServer("0.0.0.0", self.port, on_data_receive=self.handler)
                self.server.serve_forever()
            except Exception as e:
                if e.errno==98:
                    self.port+=1
                    continue
                else:
                    print(e)
                    del self
                    break
            else:
                break

    def __init__(self, callback=None):
        self.callback=callback

        global WebSocketServer, HTML, display, Thread
        from websock import WebSocketServer
        from IPython.display import HTML, display
        from threading import Thread

        self.server_thread=Thread(target=self._serve)
        self.server_thread.daemon=True
        self.server_thread.start()

    def __call__(self):
        return self.js

    def show(self):
        display(self.js)


def get_Control():
    if _cat==0 or _cat==1 or _cat==3 or _cat==5 or _cat==6:
        return AutoCar()
    elif _cat==2 or _cat==4: 
        return SerBot()

''' Artificial Intelligence '''


from .__init__ import Camera
import traitlets, os, glob, cv2, PIL.Image
from IPython.display import display
import numpy as np
from threading import Thread
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import tensorrt as trt
import atexit

def bgr8_to_jpeg(value):
    return bytes(cv2.imencode('.jpg', value)[1])

class _cam_based_class:
    _camera=None

    def __init__(self, camera=None):
        if hasattr(camera,'code') and camera.code==Camera.code:
            self._camera=camera
        else:
            print('This Camera class is not available.')
            del self
    @property
    def camera(self):
        return self._camera
    
    @camera.setter
    def camera(self,camera):
        if type(camera)==Camera:
            self._camera=camera
        else:
            print('Not available class.')

class Data_Collector(_cam_based_class):
    blocked_dir = 'collision_dataset/blocked'
    free_dir = 'collision_dataset/free'
    track_dir = 'track_dataset/'
    separator=None
    free_button=None
    blocked_button=None
    free_count=None
    blocked_count=None
    joystick=None
    image=None
    imageWidget=None
    cameraWidget=None
    toggleWidget=None
    ac=get_Control()
    max_speed=60
    min_speed=20
    fps=5
    last_cp=0
    is_ready=False
    jsave=False

    def __init__(self, separator, camera=None, auto_ready=True, save_per_sec=5):
        if camera is not None:
            super().__init__(camera)

        self.fps=save_per_sec

        if separator==Collision_Avoid or separator==Track_Follow :
            self.separator=separator
        elif type(separator)==Collision_Avoid or type(separator)==Track_Follow :
            self.separator=type(separator)
        elif separator=="Collision_Avoid":
            self.separator=Collision_Avoid
        elif separator=="Track_Follow":
            self.separator=Track_Follow
        else:
            print('Not available class. Please input a Collision_Avoid class or Track_Follow class in the 2nd parameter.')
            del self

        if auto_ready:
            self.ready()

    def __call__(self):
        self.show()

    def __del__(self):
        if self.joystick is not None:
            del self.joystick

    def _joystick_save_onclick(self, e):
        if e['new']:
            e.owner.button_style='success'
            self.jsave=True
        else:
            e.owner.button_style='danger'
            self.jsave=False

    ''' only for Collision_Avoid '''
    def save_snapshot(self,directory):
        ctime=time.strftime('%Y-%m-%d %H:%M:%S.', time.localtime(time.time()))+str(int(time.time()*100%100))
        image_path = os.path.join(directory, ctime + '.jpg')
        with open(image_path, 'wb') as f:
            f.write(self.camera.image.value)

    def save_free(self):
        self.save_snapshot(self.free_dir)
        self.free_count.value = len(os.listdir(self.free_dir))
        
    def save_blocked(self):
        self.save_snapshot(self.blocked_dir)
        self.blocked_count.value = len(os.listdir(self.blocked_dir))

    def control_as_joystick(self, value):
        if value['sep'] == "j":
            self.ac.steering=value['x']
            speed=value['y']*(self.max_speed-self.min_speed)

            if speed>0:
                self.ac.forward(speed+self.min_speed)
            elif speed<0:
                self.ac.backward(-speed+self.min_speed)
            else:
                self.ac.stop()
    ''' only for Collision_Avoid '''

    _y_scale=1/3

    ''' only for Track_Follow '''
    def save_as_joystick(self, value):
        self.image=self.camera.value
        tmp_img=bgr8_to_jpeg(self.image)

        x=(value['x']+1)/2*self.camera.width
        y=(value['y']+1)/2*self.camera.height
        
        if value['sep'] == "j":
            y=((value['y']+1)*self._y_scale)/2*self.camera.height
            y=self.camera.height*self._y_scale*2-y
        
            self.ac.steering=value['x']
            speed=value['y']*(self.max_speed-self.min_speed)

            if speed>0:
                self.ac.forward(speed+self.min_speed)
            elif speed<0:
                self.ac.backward(-speed+self.min_speed)
            else:
                self.ac.stop()
            
        
        if self.jsave or value['sep'] == "c":
            if time.time()-self.last_cp >= 1/self.fps:
                
                cv2.circle(self.image, (int(x), int(y)), 6, (0, 255, 0), 2)
                self.imageWidget.value=bgr8_to_jpeg(self.image)
                ctime=time.strftime('%Y-%m-%d %H:%M:%S.', time.localtime(time.time()))+str(int(time.time()*100000%100000))
                image_path = os.path.join(self.track_dir, str(int(x))+"_"+str(int(y))+"_"+ctime + '.jpg')
                with open(image_path, 'wb') as f:
                    f.write(tmp_img)
                    
    ''' only for Track_Follow '''

    def ready(self):
        if self.separator==Collision_Avoid:
            try:
                os.makedirs(self.free_dir)
                os.makedirs(self.blocked_dir)
            except FileExistsError:
                pass

            button_layout = widgets.Layout(width='128px', height='64px')
            self.free_button = widgets.Button(description='add free', button_style='success', layout=button_layout)
            self.blocked_button = widgets.Button(description='add blocked', button_style='danger', layout=button_layout)
            self.free_count = widgets.IntText(layout=button_layout, value=len(os.listdir(self.free_dir)))
            self.blocked_count = widgets.IntText(layout=button_layout, value=len(os.listdir(self.blocked_dir)))

            self.free_button.on_click(lambda x: self.save_free())
            self.blocked_button.on_click(lambda x: self.save_blocked())
            self.joystick=joystick(callback=self.control_as_joystick)
        elif self.separator==Track_Follow:
            try:
                os.makedirs(self.track_dir)
            except FileExistsError:
                pass

            self.imageWidget=widgets.Image(format='jpeg', width=self.camera.width, height=self.camera.height)
            self.cameraWidget=widgets.Image(format='jpeg', width=self.camera.width, height=self.camera.height)
            self.cameraWidget.add_class("clickable_image_box")
            self.camera_link = traitlets.dlink((self.camera.image, 'value'), (self.cameraWidget, 'value'))
            self.camera_link.link()
            self.toggleWidget=widgets.ToggleButton(
                value=False,
                description='Auto Collect',
                disabled=False,
                tooltip='Automatically collects datasets during joystick control.',
                button_style='danger'
            )
            self.toggleWidget.observe(self._joystick_save_onclick,'value')
            self.joystick=joystick(callback=self.save_as_joystick)
            

        self.is_ready=True

    def show(self):
        if self.camera is None:
            print('Please set camera class.')
            return

        if not self.is_ready:
            self.ready()

        if self.separator==Collision_Avoid:
            self.camera.show()
            display(widgets.HBox([self.free_count, self.free_button]))
            display(widgets.HBox([self.blocked_count, self.blocked_button]))
            display(self.joystick())
        elif self.separator==Track_Follow:
            self.imageWidget.value=bgr8_to_jpeg(self.camera.value)
            display(widgets.HBox([self.cameraWidget, self.imageWidget]))
            display(self.toggleWidget)
            display(self.joystick())
            display(HTML('<script>var list=document.getElementsByClassName("clickable_image_box");for (var i=0; i<list.length; i++) {    list[i].onclick=function(e){		var x=e.offsetX/150-1;		var y=e.offsetY/150-1;		sock_'+self.joystick.id+'.send("c,"+x+","+y);	};}</script>'))

class Collision_Avoid(_cam_based_class):
    MODEL_PATH = 'collision_avoid_model.pth'
    dataset_path = 'collision_dataset'
    datasets=None
    train_dataset=None
    test_dataset=None
    train_loader=None
    test_loader=None
    device=None
    model=None
    ready2show=False
    slider=None
    STAT_DEFINED=0
    STAT_READY=1
    _stat=STAT_DEFINED
    BATCH_SIZE=8

    mean = 255.0 * np.array([0.485, 0.456, 0.406])
    stdev = 255.0 * np.array([0.229, 0.224, 0.225])

    normalize = torchvision.transforms.Normalize(mean, stdev)

    def _preprocess(self, value):
        x = cv2.cvtColor(value, cv2.COLOR_BGR2RGB)
        x = x.transpose((2, 0, 1))
        x = torch.from_numpy(x).float()
        x = self.normalize(x)
        x = x.to(self.device)
        x = x[None, ...]
        return x

    def load_datasets(self, path=dataset_path):
        datasets_noti_widget = widgets.Label(value="Loading datasets...")
        model_noti_widget = widgets.Label(value="Creating a new model...")

        display(datasets_noti_widget)

        if not os.path.exists(path):
            print(path," doesn't exist.")
            return

        self.datasets = datasets.ImageFolder(
            path,
            transforms.Compose([
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                transforms.Resize((self.camera.width, self.camera.height)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        )

        if len(self.datasets)<=0:
            datasets_noti_widget.value="Can't access datasets. Check file permission or existence."
            return
        else:
            datasets_noti_widget.value="Load "+str(len(self.datasets))+" datasets completed."

        train_amount=int(len(self.datasets)*0.9)
        test_amount=len(self.datasets)-int(len(self.datasets)*0.9)

        self.train_dataset, self.test_dataset = torch.utils.data.random_split(self.datasets, [train_amount, test_amount])

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=4
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=4
        )

        if self.model is None:
            display(model_noti_widget)
            self.model = models.alexnet(pretrained=True)
            self.model.classifier[6] = torch.nn.Linear(self.model.classifier[6].in_features, 2)
            self.device = torch.device('cuda')
            self.model = self.model.to(self.device)
            model_noti_widget.value="Model creation completed."
            
        self._stat=self.STAT_READY

    def train(self, times=5, autosave=True):
        if self._stat < self.STAT_READY :
            print("Please load datasets as load_datasets() method.")
            return

        totally_progress_widget = widgets.FloatProgress(min=0.0, max=100.0, description='Total')
        progress_widget = widgets.FloatProgress(min=0.0, max=100.0, description='This step')
        total_percentage_widget = widgets.Label(value="0%")
        current_percentage_widget = widgets.Label(value="0%")
        remaining_time_widget = widgets.Label(value="0")
        loss_widget = widgets.Label(value="0")

        row1=widgets.HBox([totally_progress_widget, total_percentage_widget])
        row2=widgets.HBox([progress_widget, current_percentage_widget])
        row3=widgets.HBox([widgets.Label(value="Remaining"), remaining_time_widget, widgets.Label(value="sec")])
        row4=widgets.HBox([widgets.Label(value="Loss : "), loss_widget])

        display(widgets.VBox([row1, row2, row3, row4]))

        acc = 0.0

        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        
        spent_time=0

        for epoch in range(times):
            i=0

            for images, labels in iter(self.train_loader):
                time_check=time.time()

                images = images.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()

                outputs = self.model(images)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()

                i+=len(images)

                loss_widget.value=str(float(loss))
                
                cur_per = round(i / len(self.datasets) *100, 1)
                total_per = round((epoch/times + (i/len(self.datasets))/times)*100,1)

                current_percentage_widget.value=str(cur_per)+"%"
                progress_widget.value=cur_per

                total_percentage_widget.value=str(total_per)+"%"
                totally_progress_widget.value=total_per

                spent_time+=time.time()-time_check
                remaining_time_widget.value=str(int(spent_time*(100.0/total_per)-spent_time))
            
            test_error_count = 0.0
            for images, labels in iter(self.test_loader):

                time_check=time.time()
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                test_error_count += float(torch.sum(torch.abs(labels - outputs.argmax(1))))

                i+=len(images)
                
                cur_per = round(i / len(self.datasets) *100, 1)
                total_per = round((epoch/times + (i/len(self.datasets))/times)*100,1)

                current_percentage_widget.value=str(cur_per)+"%"
                progress_widget.value=cur_per

                total_percentage_widget.value=str(total_per)+"%"
                totally_progress_widget.value=total_per

                spent_time+=time.time()-time_check
                remaining_time_widget.value=str(int(spent_time*(100.0/total_per)-spent_time))

            time_check=time.time()
            cur_per = round(i / len(self.datasets) *100, 1)
            current_percentage_widget.value=str(cur_per)+"%"
            progress_widget.value=cur_per

            total_per = round((epoch/times + (i/len(self.datasets))/times)*100,1)
            total_percentage_widget.value=str(total_per)+"%"
            totally_progress_widget.value=total_per

            tmp_acc = round((1.0 - float(test_error_count) / float(len(self.test_dataset)))*100,1)
            print(str(epoch),' step accuracy : '+str(tmp_acc)+"%")
            if tmp_acc > acc:
                if autosave:
                    torch.save(self.model.state_dict(), self.MODEL_PATH)
                acc = tmp_acc

            spent_time+=time.time()-time_check

        total_per=100
        total_percentage_widget.value=str(total_per)+"%"
        totally_progress_widget.value=total_per

        remaining_time_widget.value="0"

    def load_model(self,path=MODEL_PATH):
        if not os.path.exists(path):
            print(path," doesn't exist.")
            return

        self.model = torchvision.models.alexnet(pretrained=True)
        self.model.classifier[6] = torch.nn.Linear(self.model.classifier[6].in_features, 2)
        self.model.load_state_dict(torch.load(path))
        self.device = torch.device('cuda')
        self.model = self.model.to(self.device)
        self._stat=self.STAT_READY

    def save_model(self,path=MODEL_PATH):
        if self.model is not None:
            torch.save(self.model.state_dict(), path)
            print("Save completed.")
        else:
            print("The model can't be saved cause it's None.")

    def show(self):
        if self.slider is None:
            self.slider = widgets.FloatSlider(description='blocked', min=0.0, max=1.0, orientation='vertical')
        display(widgets.HBox([self.camera.image, self.slider]))

    def run(self, value=None, callback=None):
        if self.model is None :
            print("Please load datasets as load_datasets() method or load a trained model as load_model() method.")
            return

        x = self.camera.value if value is None else value
        x = self._preprocess(x)
        y = self.model(x)
        
        y = F.softmax(y, dim=1)
        
        prob_blocked = float(y.flatten()[0])
        
        if self.slider is not None:
            self.slider.value = prob_blocked
        
        try:
            if callback is not None:
                callback(prob_blocked)
        except Exception as e:
            print("Error : Can't callback this method.")
            print(e)

        return prob_blocked

class Object_Follow(_cam_based_class):
    model=None
    image=None
    value=None
    label_list=["background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", None, "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", None, "backpack", "umbrella", None, None, "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", None, "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", None, "dining table", None, None, "toilet", None, "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", None, "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    def __init__(self, camera, classes_path=None):
        if _cat==4 or _cat==5 or _cat==6:
            self.default_path=os.path.abspath(__file__+"/../model/yolov4-tiny/")

            super().__init__(camera)
            self.image = widgets.Image(format='jpeg', width=camera.width, height=camera.height)
            global YOLOv4
            from yolov4.tf import YOLOv4
            self.model = YOLOv4(tiny=True)

            if classes_path is None : classes_path=self.default_path+"/coco.names"
            self.model.classes = classes_path

            self.model.input_size = (camera.width, camera.height)
        else:
            self.default_path=os.path.abspath(__file__+"/../model/SSD_MobileNet_COCO_trt/")
            super().__init__(camera)
            self.image = widgets.Image(format='jpeg', width=camera.width, height=camera.height)

    def load_model(self,path=None):
        if _cat==4 or _cat==5 or _cat==6:
            if path is None:
                path=self.default_path+'/yolov4-tiny.weights'

            try:
                self.model.make_model()
                self.model.load_weights(path, weights_type="yolo")
            except Exception as e:
                if e.errno==2:
                    print("Can't find pre-trained model file. Please refer https://github.com/hanback-docs/yolov4-tiny")
                else:
                    print(e)
        else:
            if path is None:
                path=self.default_path+'/ssd_mobilenet_v2_coco.engine'

            try:
                self.model = ObjectDetector(path)
            except Exception as e:
                if e.errno==2:
                    print("Can't find pre-trained model file. Please refer https://github.com/hanback-docs/ssd_mobilenet_v2_coco_engine.")
                else:
                    print(e)

    def show(self):
        display(self.image)

    def detect(self, image=None, index=None, threshold=0.5, show=True, callback=None):
        if image is None:
            image=self.camera.value

        if _cat==4 or _cat==5 or _cat==6:
            self.label_list=list(self.model.classes.values())

        if type(index)==str:
            try:
                index=self.label_list.index(index)
            except ValueError:
                print("index is not available.")
                return

        width=self.camera.width
        height=self.camera.height
        if self.model is None:
            print("Please load a object detector model as load_model() method.")
        else:
            detections=[]

            if _cat==4 or _cat==5 or _cat==6:
                detections=self.model.predict(image, score_threshold=threshold)

                detections=[[{'label':int(det[4]), 'confidence':det[5], 'bbox':[det[0]-det[2]/2,
                                                                                det[1]-det[3]/2,
                                                                                det[0]+det[2]/2,
                                                                                det[1]+det[3]/2]} 
                            for det in detections if det[5]>=threshold]]
            else:
                detections=self.model(image)
                detections=[[det for det in detections[0] if det['confidence']>=threshold]]
                
            result=detections

            for det in detections[0]:
                bbox = det['bbox']
                cv2.rectangle(image, (int(width * bbox[0]), int(height * bbox[1])), (int(width * bbox[2]), int(height * bbox[3])), (255, 0, 0), 2)

                label=self.label_list[det['label']]
                label_size=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)[0]

                cv2.rectangle(image, (int(width * bbox[0]), int(height * bbox[1]-label_size[1]-5)), (int(width * bbox[0]+label_size[0]), int(height * bbox[1])), (255, 0, 0), -1)
                cv2.putText(image, label, (int(width * bbox[0]), int(height * bbox[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            if index is not None:
                matching_detections = [d for d in detections[0] if d['label'] == int(index)]

                det=None
                max_bbox_size=0
                
                for d in matching_detections:
                    cur_size=(width*d['bbox'][0]-width*d['bbox'][2])**2+(height*d['bbox'][1]-height*d['bbox'][3])**2
                    if max_bbox_size<cur_size :
                        max_bbox_size=cur_size
                        det = d
                
                if det is not None:
                    bbox = det['bbox']
                    label=self.label_list[det['label']]
                    label_size=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)[0]
                    cv2.rectangle(image, (int(width * bbox[0]), int(height * bbox[1])), (int(width * bbox[2]), int(height * bbox[3])), (0, 255, 0), 4)
                    cv2.rectangle(image, (int(width * bbox[0]-2), int(height * bbox[1]-label_size[1]-5)), (int(width * bbox[0]+label_size[0]), int(height * bbox[1])), (0, 255, 0), -1)
                    cv2.putText(image, label, (int(width * bbox[0]), int(height * bbox[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    x = (bbox[0] + bbox[2]) / 2.0 - 0.5
                    y = (bbox[1] + bbox[3]) / 2.0 - 0.5
                    det_size=abs(width*bbox[0]-width*bbox[2])*abs(height*bbox[1]-height*bbox[3])
                    cam_size=width*height
                    size=det_size/cam_size
                    result={'x':x, 'y':y, 'size_rate':size}
                else:
                    result=None

            if show:
                self.image.value=bgr8_to_jpeg(image)
                self.value=image

            if callback is not None:
                callback(result)

            return result


class Track_Follow(_cam_based_class):
    MODEL_PATH = 'track_follow_model.pth'
    dataset_path = 'track_dataset'
    BATCH_SIZE = 8
    model=None
    device=None
    datasets=None
    optimizer=None
    prob=None
    probWidget=None
    STAT_DEFINED=0
    STAT_READY=1
    _stat=STAT_DEFINED

    def __init__(self,camera):
        super().__init__(camera)
        self.probWidget = widgets.Image(format='jpeg', width=camera.width, height=camera.height)

    def _preprocess(self, value):
        mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        device = torch.device('cuda')
        value = PIL.Image.fromarray(value)
        value = transforms.functional.to_tensor(value).to(device)
        value.sub_(mean[:, None, None]).div_(std[:, None, None])
        return value[None, ...]

    def load_datasets(self, path=dataset_path):
        datasets_noti_widget = widgets.Label(value="Loading datasets...")
        model_noti_widget = widgets.Label(value="Creating a new model...")

        display(datasets_noti_widget)

        if not os.path.exists(path):
            print(path," doesn't exist.")
            return

        TRANSFORMS = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
            transforms.Resize((self.camera.width, self.camera.height)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.datasets = XYDataset(path, TRANSFORMS, random_hflip=True)

        if len(self.datasets)<=0:
            datasets_noti_widget.value="Can't access datasets. Check file permission or existence."
            return
        else:
            datasets_noti_widget.value="Load "+str(len(self.datasets))+" datasets completed."

        if self.model is None:
            display(model_noti_widget)
            self.model = torchvision.models.resnet18(pretrained=True)
            self.model.fc = torch.nn.Linear(512, 2)
            self.device = torch.device('cuda')
            self.model = self.model.to(self.device)
            model_noti_widget.value="Model creation completed."

        self.dataset_path=path
        self._stat=self.STAT_READY

    def train(self, times=5, autosave=True):
        if self._stat < self.STAT_READY:
            print("Please load the datasets as load_datasets() method.")
            return

        totally_progress_widget = widgets.FloatProgress(min=0.0, max=100.0, description='Total')
        progress_widget = widgets.FloatProgress(min=0.0, max=100.0, description='This step')
        total_percentage_widget = widgets.Label(value="0%")
        current_percentage_widget = widgets.Label(value="0%")
        remaining_time_widget = widgets.Label(value="0")
        loss_widget = widgets.Label(value="0")

        row1=widgets.HBox([totally_progress_widget, total_percentage_widget])
        row2=widgets.HBox([progress_widget, current_percentage_widget])
        row3=widgets.HBox([widgets.Label(value="Remaining"), remaining_time_widget, widgets.Label(value="sec")])
        row4=widgets.HBox([widgets.Label(value="Loss : "), loss_widget])

        display(widgets.VBox([row1, row2, row3, row4]))

        self.optimizer = torch.optim.Adam(self.model.parameters())
    
        try:
            train_loader = torch.utils.data.DataLoader(
                self.datasets,
                batch_size=self.BATCH_SIZE,
                shuffle=True
            )

            self.model = self.model.train()

            spent_time=0

            for n in range(times):
                i = 0
                sum_loss = 0.0
                error_count = 0.0
                for images, xy in iter(train_loader):
                    time_check=time.time()

                    images = images.to(self.device)
                    xy = xy.to(self.device)

                    self.optimizer.zero_grad()

                    outputs = self.model(images)

                    loss = 0.0
                    for m in range(len(images)):
                        loss += torch.mean((outputs[m][0:2] - xy[m])**2)
                    loss /= len(images)

                    loss.backward()
                    self.optimizer.step()

                    count = len(images)
                    i += count
                    sum_loss += float(loss)
                    loss_widget.value=str(float(loss))
                    cur_per = round(i / len(self.datasets) *100, 1)
                    total_per = round((n/times + (i/len(self.datasets))/times)*100,1)

                    current_percentage_widget.value=str(cur_per)+"%"
                    progress_widget.value=cur_per

                    total_percentage_widget.value=str(total_per)+"%"
                    totally_progress_widget.value=total_per
                    
                    spent_time+=time.time()-time_check
                    remaining_time_widget.value=str(int(spent_time*(100.0/total_per)-spent_time))
                    
                time_check=time.time()
                cur_per = round(i / len(self.datasets) *100, 1)
                current_percentage_widget.value=str(cur_per)+"%"
                progress_widget.value=cur_per

                total_per = round((n/times + (i/len(self.datasets))/times)*100,1)
                total_percentage_widget.value=str(total_per)+"%"
                totally_progress_widget.value=total_per

                _loss = sum_loss / i
                print(n,"step loss : ", _loss)

                spent_time+=time.time()-time_check

            total_per=100
            total_percentage_widget.value=str(total_per)+"%"
            totally_progress_widget.value=total_per

            remaining_time_widget.value="0"

        except Exception as e:
            print(e)
            pass
        
        if autosave:
            torch.save(self.model.state_dict(), self.MODEL_PATH)
        self.model = self.model.eval()

    def show(self):
        display(self.probWidget)

    def run(self, value=None, callback=None):
        if self.model is None :
            print("Please load datasets as load_datasets() method or load a trained model as load_model() method.")
            return

        self.prob = self.camera.value.copy()
        preprocessed = self._preprocess(self.prob)
        output = self.model(preprocessed).detach().cpu().numpy().flatten()
        x = output[0]
        y = output[1]
        
        cX = int(self.camera.width * (x / 2.0 + 0.5))
        cY = int(self.camera.height * (y / 2.0 + 0.5))
        
        self.prob = cv2.circle(self.prob, (cX, cY), 6, (255, 255, 255), 2)
        self.probWidget.value = bgr8_to_jpeg(self.prob)

        result={"x":x,"y":y}

        if callback is not None:
            callback(result)

        return result

    def load_model(self, path=MODEL_PATH):
        if not os.path.exists(path):
            print(path," doesn't exist.")
            return
            
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = torch.nn.Linear(512, 2)
        self.model.load_state_dict(torch.load(path))
        self.device = torch.device('cuda')
        self.model = self.model.to(self.device)
        
        print("Load completed.")
        
    def save_model(self, path=MODEL_PATH):
        if self.model is not None:
            torch.save(self.model.state_dict(), path)
            print("Save completed.")
        else:
            print("The model can't be saved cause it's None.")


class XYDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform=None, random_hflip=False):
        super(XYDataset, self).__init__()
        self.directory = directory
        self.transform = transform
        self.refresh()
        self.random_hflip = random_hflip
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image = cv2.imread(ann['image_path'], cv2.IMREAD_COLOR)
        image = PIL.Image.fromarray(image)
        width = image.width
        height = image.height
        if self.transform is not None:
            image = self.transform(image)
        
        x = 2.0 * (ann['x'] / width - 0.5) # -1 left, +1 right
        y = 2.0 * (ann['y'] / height - 0.5) # -1 top, +1 bottom
        
        if self.random_hflip and float(np.random.random(1)) > 0.5:
            image = torch.from_numpy(image.numpy()[..., ::-1].copy())
            x = -x
            
        return image, torch.Tensor([x, y])
    
    def _parse(self, path):
        basename = os.path.basename(path)
        items = basename.split('_')
        x = items[0]
        y = items[1]
        return int(x), int(y)
        
    def refresh(self):
        self.annotations = []
        for image_path in glob.glob(os.path.join(self.directory, '*.jpg')):
            x, y = self._parse(image_path)
            self.annotations += [{
                'image_path': image_path,
                'x': x,
                'y': y
            }]

#---------------------------------------------  Object Recognition TRT Model for OF ---------------------------------------------


import ctypes
import subprocess
import tensorrt as trt

TRT_INPUT_NAME = 'input'
TRT_OUTPUT_NAME = 'nms'
LABEL_IDX = 1
CONFIDENCE_IDX = 2
X0_IDX = 3
Y0_IDX = 4
X1_IDX = 5
Y1_IDX = 6

def parse_boxes(outputs):
    bboxes = outputs[0]
            
    all_detections = []
    for i in range(bboxes.shape[0]):

        detections = []
        for j in range(bboxes.shape[2]):

            bbox = bboxes[i][0][j]
            label = bbox[LABEL_IDX]

            if label < 0: 
                break

            detections.append(dict(
                label=int(label),
                confidence=float(bbox[CONFIDENCE_IDX]),
                bbox=[
                    float(bbox[X0_IDX]),
                    float(bbox[Y0_IDX]),
                    float(bbox[X1_IDX]),
                    float(bbox[Y1_IDX])
                ]
            ))

        all_detections.append(detections)

    return all_detections

def load_plugins():
    library_path = os.path.join(
        os.path.dirname(__file__), 'libssd_tensorrt.so')
    ctypes.CDLL(library_path)

def torch_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return torch.int8
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError('%s is not supported by torch' % dtype)

def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device('cuda')
    elif device == trt.TensorLocation.HOST:
        return torch.device('cpu')
    else:
        return TypeError('%s is not supported by torch' % device)

class TRTModel(object):
    
    def __init__(self, engine_path, input_names=None, output_names=None, final_shapes=None):
        
        # load engine
        self.logger = trt.Logger()
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        
        if input_names is None:
            self.input_names = self._trt_input_names()
        else:
            self.input_names = input_names
            
        if output_names is None:
            self.output_names = self._trt_output_names()
        else:
            self.output_names = output_names
            
        self.final_shapes = final_shapes
        
        # destroy at exit
        atexit.register(self.destroy)
    
    def _input_binding_indices(self):
        return [i for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i)]
    
    def _output_binding_indices(self):
        return [i for i in range(self.engine.num_bindings) if not self.engine.binding_is_input(i)]
    
    def _trt_input_names(self):
        return [self.engine.get_binding_name(i) for i in self._input_binding_indices()]
    
    def _trt_output_names(self):
        return [self.engine.get_binding_name(i) for i in self._output_binding_indices()]
    
    def create_output_buffers(self, batch_size):
        outputs = [None] * len(self.output_names)
        for i, output_name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            if self.final_shapes is not None:
                shape = (batch_size, ) + self.final_shapes[i]
            else:
                shape = (batch_size, ) + tuple(self.engine.get_binding_shape(idx))
            device = torch_device_from_trt(self.engine.get_location(idx))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[i] = output
        return outputs
    
    def execute(self, *inputs):
        batch_size = inputs[0].shape[0]
        
        bindings = [None] * (len(self.input_names) + len(self.output_names))
        
        # map input bindings
        inputs_torch = [None] * len(self.input_names)
        for i, name in enumerate(self.input_names):
            idx = self.engine.get_binding_index(name)
            
            # convert to appropriate format
            inputs_torch[i] = torch.from_numpy(inputs[i])
            inputs_torch[i] = inputs_torch[i].to(torch_device_from_trt(self.engine.get_location(idx)))
            inputs_torch[i] = inputs_torch[i].type(torch_dtype_from_trt(self.engine.get_binding_dtype(idx)))
            
            bindings[idx] = int(inputs_torch[i].data_ptr())
            
        output_buffers = self.create_output_buffers(batch_size)
        
        # map output bindings
        for i, name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(name)
            bindings[idx] = int(output_buffers[i].data_ptr())
        
        self.context.execute(batch_size, bindings)
        
        outputs = [buffer.cpu().numpy() for buffer in output_buffers]
                                 
        return outputs
    
    def __call__(self, *inputs):
        return self.execute(*inputs)

    def destroy(self):
        self.runtime.destroy()
        self.logger.destroy()
        self.engine.destroy()
        self.context.destroy()

mean = 255.0 * np.array([0.5, 0.5, 0.5])
stdev = 255.0 * np.array([0.5, 0.5, 0.5])


def bgr8_to_ssd_input(camera_value):
    x = camera_value
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.transpose((2, 0, 1)).astype(np.float32)
    x -= mean[:, None, None]
    x /= stdev[:, None, None]
    return x[None, ...]

class ObjectDetector(object):
    
    def __init__(self, engine_path, preprocess_fn=bgr8_to_ssd_input):
        logger = trt.Logger()
        trt.init_libnvinfer_plugins(logger, '')
        load_plugins()
        self.trt_model = TRTModel(engine_path, input_names=[TRT_INPUT_NAME],
                                  output_names=[TRT_OUTPUT_NAME, TRT_OUTPUT_NAME + '_1'])
        self.preprocess_fn = preprocess_fn
        
    def execute(self, *inputs):
        trt_outputs = self.trt_model(self.preprocess_fn(*inputs))
        return parse_boxes(trt_outputs)
    
    def __call__(self, *inputs):
        return self.execute(*inputs)
