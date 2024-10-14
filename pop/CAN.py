import can
import time
import __main__
from threading import Thread, Lock

lock=Lock()

__main__._ultraData = [0,0,0,0,0,0]
__main__._psdData = [0,0,0]
__main__._alarm = [0,0,0,0,0,0,0,0,0]
__main__._front_ultraData = [0,0,0,0,0]
__main__._back_ultraData = [0,0,0,0,0]

class Can:
    def __init__(self):
        self._bus = can.interface.Bus(channel='can0', bustype='socketcan_native')
    
    def __del__(self):
        self._bus = 0

    def write(self, msg_id, buf, is_extended=False):
        msg = can.Message(arbitration_id=msg_id,is_extended_id=is_extended,data=buf)
        try:
            self._bus.send(msg)
            time.sleep(0.005)
        except can.CanError:
            print("Can Interface Message does not Send")

    def read(self,timeOut=2):
        recv_buf = self._bus.recv(timeout=timeOut)
        if recv_buf:
            return recv_buf

    def setFilter(self, can_id, mask):
        self._bus.set_filters([{"can_id": can_id, "can_mask": mask, "extended": False}])

class OmniWheel:
    MOTOR_CONTROL       =   0x101
    MOTOR_STOP          =   0x102
    SENSOR_REQUEST      =   0x103
    OBSTACLE_DISTANCE   =   0x104
    OBSTACLE_DETECT     =   0x133
    OBSTACLE_ALARM      =   0x134
    RECV_ULTRASONIC     =   0x105
    RECV_PSD            =   0x106
    ULTRASONIC          =   0x001
    PSD                 =   0x002
    ALARM               =   0x003
    SENSOR_ALL          =   0x004

    def __init__(self):
        super().__init__()
        self._ultraEnable = 0x0
        self._psdEnable = 0x0
        self._func = None
        self._param = None
        self._wheels = [0,0,0,0,0,0]

        self.can=Can()
        #self.can.setFilter(0x130,0x7F0)
        self.allSensorEnable()
        self.readStart()

    def __del__(self):
        self.readStop()
        super().__del__()

    def wheel(self, id, value):
        if id == 1 :
            if value < 0 :
                self._wheels[0]=int(abs(value))
                self._wheels[1]=0
            elif value > 0 :
                self._wheels[0]=0
                self._wheels[1]=int(abs(value))
            else :
                self._wheels[0]=0
                self._wheels[1]=0
        elif id == 2 :
            if value < 0 :
                self._wheels[2]=int(abs(value))
                self._wheels[3]=0
            elif value > 0 :
                self._wheels[2]=0
                self._wheels[3]=int(abs(value))
            else :
                self._wheels[2]=0
                self._wheels[3]=0
        elif id == 3 :
            if value < 0 :
                self._wheels[4]=int(abs(value))
                self._wheels[5]=0
            elif value > 0 :
                self._wheels[4]=0
                self._wheels[5]=int(abs(value))
            else :
                self._wheels[4]=0
                self._wheels[5]=0
        time.sleep(0.01)
        self.can.write(self.MOTOR_CONTROL,self._wheels)

    def forward(self, data=None):
        if data is None:
            self.can.write(self.MOTOR_CONTROL,self._wheels)
        else:
            self.can.write(self.MOTOR_CONTROL,[data[0],0,data[1],0,data[2],0])

    def backward(self, data=None):
        if data is None:
            tmp=self._wheels[:]
            tmp=tmp[0:2][::-1]+tmp[2:4][::-1]+tmp[4:6][::-1]
            self.can.write(self.MOTOR_CONTROL,tmp)
        else:
            self.can.write(self.MOTOR_CONTROL,[0,data[0],0,data[1],0,data[2]])

    def setObstacleRange(self, ultra_distance, psd_distance):
        self.can.write(self.OBSTACLE_DISTANCE,[ultra_distance,psd_distance,0,0,0,0])

    def stop(self):
        self.can.write(self.MOTOR_STOP,[0,0,0,0,0,0])

    def allSensorEnable(self):
        self._ultraEnable = 0x3F
        self._psdEnable = 0x7
        self.can.write(self.SENSOR_REQUEST,[self._ultraEnable,self._psdEnable])

    def ultraEnable(self,enable=[1,1,1,1,1,1]):
        for i in range(6):
            if enable[i] == 1:
                self._ultraEnable = self._ultraEnable | (0x1 << i) 
            elif enable[i] == 0:
                self._ultraEnable = self._ultraEnable & ~(0x1 << i) 
            else:
                pass

        self.can.write(self.SENSOR_REQUEST,[self._ultraEnable,self._psdEnable])

    def psdEnable(self,enable=[1,1,1]):
        for i in range(3):
            if enable[i] == 1:
                self._psdEnable = self._psdEnable | (0x1 << i) 
            elif enable[i] == 0:
                self._psdEnable = self._psdEnable & ~(0x1 << i) 
            else:
                pass

        self.can.write(self.SENSOR_REQUEST,[self._ultraEnable,self._psdEnable])

    def getEnable(self):
        return self._ultraEnable, self._psdEnable

    def sensorDisable(self):
        self._ultraEnable = 0x0
        self._psdEnable = 0x0
        self.can.write(self.SENSOR_REQUEST,[self._ultraEnable,self._psdEnable])

    def _readSensor(self):
        while True:
            recv_data = self.can.read()
            lock.acquire()
            try:
                if recv_data.arbitration_id == self.RECV_ULTRASONIC:
                    for i in range(6):
                        __main__._ultraData[i] = int(recv_data.data[i])
                elif recv_data.arbitration_id == self.RECV_PSD:
                    for i in range(3):
                        __main__._psdData[i] = int(recv_data.data[i])
                elif recv_data.arbitration_id == self.OBSTACLE_ALARM and recv_data.dlc == 2:
                    self._alarm[0] = 1
                    for i in range(2):
                        __main__._alarm[i+1] = int(recv_data.data[i])
                elif recv_data.arbitration_id == self.OBSTACLE_DETECT or recv_data.dlc == 8:
                    self._alarm[0] = 2 
                    for i in range(8):
                        __main__._alarm[i+1] = int(recv_data.data[i])

            except: 
                pass
            lock.release()
            time.sleep(0.1)

    def readStart(self):
        if not hasattr(__main__, 'read_thread') or not __main__.read_thread.isAlive():
            __main__.read_thread = Thread(target=self._readSensor)
            __main__.read_thread.start()

    def readStop(self):
        if hasattr(self, 'thread'):
            __main__.read_thread.join()

    def read(self,msgType=SENSOR_ALL):
        lock.acquire()
        v=None
        if msgType == self.ULTRASONIC:
            v=__main__._ultraData
        elif msgType == self.PSD:
            v=__main__._psdData
        elif msgType == self.ALARM:
            v=__main__._alarm
        elif msgType == self.SENSOR_ALL:
            v=__main__._ultraData, __main__._psdData
        lock.release()
        return v

    def setCallback(self, func, param=None):
        self._func = func 
        self._param = param
        if not hasattr(self, 'callback_thread') or not self.callback_thread.isAlive():
            self.callback_thread = Thread(target=self._func)
            self.callback_thread.start() if (self._func != None) else self.callback_thread.stop()


class Car:
    steer_range=18
    FRONT_ULTRASONIC=0x135
    BACK_ULTRASONIC=0x136
    
    def __init__(self):
        self.__can=Can()
        self.ultraEnable()
        self.readStart()
        
    def wheel(self, value):
        data=[0,0]
        
        if abs(value)>100:
            value=100*abs(value)/value
            
        if value>0:
            data[0]=int(value)
        elif value<0:
            data[1]=int(abs(value))
            
        self.__can.write(0x101, data)
        
    def steer(self, value):
        if abs(value)>1:
            value=abs(value)/value
            
        data=[int(90-self.steer_range*value)]
        self.__can.write(0x102, data)
        
    def camPan(self, value):
        if abs(value)>60:
            value=60*abs(value)/value
            
        data=[int(90-value)]
        
        self.__can.write(0x103,data)
        
    def camTilt(self, value):
        if value<0: value=0
        elif value>90: value=90
            
        value*=0.744444
        
        data=[int(23+value)]
        
        self.__can.write(0x104,data)
            
    def ultraEnable(self,enable=[[1,1,1,1,1],[1,1,1,1,1]]):
        data=[]
        for n in range(2):
            value=0
            for i in range(5):
                if enable[n][i] == 1:
                    value|=(1<<i) 
                elif enable[n][i] == 0:
                    value&=~(1<<i) 
                else:
                    pass
            data.append(value)

        self.__can.write(0x105,data)

    def ultraDisable(self):
        self.ultraEnable([[0,0,0,0,0],[0,0,0,0,0]])

    def _readSensor(self):
        while True:
            recv_data = self.__can.read()
            lock.acquire()
            try:
                if recv_data.arbitration_id == self.FRONT_ULTRASONIC:
                    __main__._front_ultraData = list(map(int, recv_data.data))
                elif recv_data.arbitration_id == self.BACK_ULTRASONIC:
                    __main__._back_ultraData = list(map(int, recv_data.data))
            except: 
                pass
            lock.release()
            time.sleep(0.1)

    def readStart(self):
        if not hasattr(__main__, 'read_thread') or not __main__.read_thread.isAlive():
            __main__.read_thread = Thread(target=self._readSensor)
            __main__.read_thread.start()

    def readStop(self):
        if hasattr(self, 'thread'):
            __main__.read_thread.join()

    def read(self):
        lock.acquire()
        v=[__main__._front_ultraData, __main__._back_ultraData]
        lock.release()
        return v