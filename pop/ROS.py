import numpy as np
import time
import rospy
import cv2
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped 
import __main__
import subprocess as sp


rospy.init_node('driver')

class SLAM:
    def __init__(self):
        sp.call(["roslaunch rplidar_ros rplidar.launch"], shell=True)
        sp.call(["roslaunch hector_slam_launch tutorial.launch"], shell=True)

        rospy.Subscriber("/map", OccupancyGrid, self._map)
        rospy.Subscriber("/slam_out_pose", PoseStamped, self._pose)

        self.map=None
        self.pose=None

    def _map(self, data):
        temp_map=np.array(data.data).reshape(2048,2048)
        
        temp_map=np.transpose(np.where(temp_map>=100,True,False))

        self.map=temp_map
        
    def _pose(self, data):
        pose=data.pose
        
        arr={}
        arr['position']={'x':-pose.position.x, 'y':pose.position.y, 'z':pose.position.z}
        arr['orientation']={'x':pose.orientation.x, 'y':pose.orientation.y, 'z':pose.orientation.z, 'w':pose.orientation.w, 'yaw':(-np.arctan(pose.orientation.z/pose.orientation.w)*2)%(np.pi*2)}
        
        self.pose=arr

class Dynamic_Window_Approach:
    max_distance=2.0
    distance_step=0.5
    max_steer=20
    steer_step=5
    threshold=0
    
    def __init__(self, car_width=None, car_length=None, speed=0.2, map_size=(2048,2048), resolution=20):
        self.width=None
        self.length=None

        if "_cat" in dir(__main__):
            if __main__._cat==0:
                self.width=0.15
                self.length=0.21
            elif __main__._cat==2:
                self.width=0.15
                self.length=0.5
            elif __main__._cat==3:
                self.width=0.268
                self.length=0.291

        if car_width is not None:
            self.width=car_width
        elif self.width is None:
            raise ValueError("No information found for this device. Please set the car_width in meters.")

        if car_length is not None:
            self.length=car_length
        elif self.length is None:
            raise ValueError("No information found for this device. Please set the car_length in meters.")

        self.map_size=np.array(map_size)
        self.resolution=resolution
        self.speed=speed
        
    def __call__(self, map, pose, direction):
        dist=np.sqrt(np.sum((np.array((pose[1],pose[0]))-self.destination)**2))
        
        if dist>=0.5:
            ret_steer=-self.max_steer
            ret_dist=float('inf')
            ret_kernel=None
            ret_colls=0
            sw=False
            if np.sum(self.map_size == np.array(map.shape[:2])) != 2: self.map_size=np.array(map.shape[:2])

            for i, distance in enumerate(np.arange(self.distance_step, self.max_distance+1, self.distance_step)[::-1]):
                for steer in np.arange(-self.max_steer, self.max_steer+1, self.steer_step):
                    colls, dist, kernel = self.check_collision(map, pose, direction, steer, distance, forward=True)
                    if colls<=self.threshold:
                        if ret_dist>dist:
                            sw=True
                            ret_steer=steer
                            ret_dist=dist
                            ret_kernel=kernel
                            ret_colls=colls

            return ret_steer if sw else None
        else:
            return None
        
    @property
    def destination(self):
        ret=(self.__dest-self.map_size/2)/self.resolution
        ret[1]*=-1
        return ret
    
    @destination.setter
    def destination(self, coord):
        if type(coord) in (tuple,list):
            tmp=np.array(coord)
            tmp[1]*=-1
            self.__dest=self.map_size/2+tmp*self.resolution
        else: raise ValueError("Please set a tuple or list.")
        
    def circular_mask(self, size, center, min_radius, max_radius, distance=1, forward=True):
        h=size[0]
        w=size[1]

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
        maskR=(dist_from_center >= min_radius) & (dist_from_center <= max_radius)
        return maskR
    
    def range_mask(self, size, center, radius, coord, theta, distance=1, forward=True, right=True):
        h=size[0]
        w=size[1]

        Y, X = np.ogrid[:h, :w]

        tc=(distance%(2*np.pi*radius))/radius
        tc=tc if forward else (-tc)%(2*np.pi)
        tc=tc if right else (-tc)%(2*np.pi)

        fA=lambda x,y:y>np.tan(max((theta%(2*np.pi)+tc)%(2*np.pi),1e-7))*(x-center[0])+center[1]
        fC=lambda x,y:y<=np.tan(max(theta%(2*np.pi),1e-7))*(x-center[0])+center[1]

        maskA=fA(X,Y)
        maskC=fC(X,Y)

        tX=(coord[0]-center[0])*np.cos(tc)-(coord[1]-center[1])*np.sin(tc)+center[0]
        tY=(coord[0]-center[0])*np.sin(tc)+(coord[1]-center[1])*np.cos(tc)+center[1]

        if not fA(coord[0],coord[1]): maskA=~maskA

        if not fC(tX,tY): maskC=~maskC

        maskR=maskA & maskC

        if tc>np.pi: maskR=~maskR
        if not forward: maskR=~maskR
        if not right: maskR=~maskR

        return maskR
    
    def path_mask(self, pose, direction, steer, distance, size, forward=True):
        if steer==0:
            tf_direction=direction if forward else (direction+2)%2-1

            w=size[0]
            h=size[1]

            Y, X = np.ogrid[:h, :w]

            mask=None

            if tf_direction==0:
                xS=-self.width*self.resolution/2+pose[0]
                xE=self.width*self.resolution/2+pose[0]
                yS=-distance+pose[1]
                yE=pose[1]

                mask=(xS<=X) & (X<=xE) & (yS<=Y) & (Y<=yE)
            elif tf_direction in [-1,1]:
                xS=-self.width*self.resolution/2+pose[0]
                xE=self.width*self.resolution/2+pose[0]
                yS=pose[1]
                yE=distance+pose[1]

                mask=(xS<=X) & (X<=xE) & (yS<=Y) & (Y<=yE)
            elif tf_direction==0.5:
                xS=pose[1]
                xE=distance+pose[1]
                yS=-self.width*self.resolution/2+pose[1]
                yE=self.width*self.resolution/2+pose[1]

                mask=(xS<=X) & (X<=xE) & (yS<=Y) & (Y<=yE)
            elif tf_direction==-0.5:
                xS=-distance+pose[1]
                xE=pose[1]
                yS=-self.width*self.resolution/2+pose[1]
                yE=self.width*self.resolution/2+pose[1]

                mask=(xS<=X) & (X<=xE) & (yS<=Y) & (Y<=yE)
            else:
                theta_x=((tf_direction-0.5)%2)*np.pi
                theta_y=(tf_direction%2)*np.pi
                xG=np.tan(theta_x)
                yG=np.tan(theta_y)

                alpha=lambda x,y:((x)*np.cos(theta_y)-(y)*np.sin(theta_y))+pose[0]
                beta=lambda x,y:((x)*np.sin(theta_y)+(y)*np.cos(theta_y))+pose[1]

                x1=lambda x:xG*(x-alpha(-self.width*self.resolution/2,0))+beta(-self.width*self.resolution/2,0)
                x2=lambda x:xG*(x-alpha(self.width*self.resolution/2,0))+beta(self.width*self.resolution/2,0)

                y1=lambda x:yG*(x-alpha(0,-distance))+beta(0,-distance)
                y2=lambda x:yG*(x-alpha(0,0))+beta(0,0)

                xS=Y>=x1(X) if tf_direction>0 else Y<=x1(X)
                xE=Y<=x2(X) if tf_direction>0 else Y>=x2(X)
                yS=Y>=y1(X) if abs(tf_direction)<0.5 else Y<=y1(X)
                yE=Y<=y2(X) if abs(tf_direction)<0.5 else Y>=y2(X)

                mask=xS & xE & yS & yE

            return mask
        else:
            tf_direction=(direction%2)*np.pi

            tf_steer=max(np.radians(steer%360),1e-2)

            rad_M=np.array((self.length/np.tan(tf_steer),self.length))*self.resolution
            rotation_kernel=np.array([[np.cos(tf_direction),-np.sin(tf_direction)],
                                     [np.sin(tf_direction),np.cos(tf_direction)]])
            rad_M=np.matmul(rotation_kernel,rad_M) + pose

            wheels=np.array([
                ((-self.width/2),0),
                ((self.width/2),0),
                ((-self.width/2),self.length),
                ((self.width/2),self.length)
            ])*self.resolution

            rads=np.sqrt(np.sum((wheels-(np.array((self.length/np.tan(tf_steer),self.length))*self.resolution))**2, axis=-1))

            min_rad=np.min(rads)
            max_rad=np.max(rads)
            avg_rad=np.average(rads)

            return self.range_mask(size,rad_M,avg_rad,pose,tf_direction,distance,forward,steer>=0) & self.circular_mask(size,rad_M,min_rad,max_rad)

    def check_collision(self, bin_map, pose, direction, steer, distance, forward=True):
        size=np.array((len(bin_map[0]),len(bin_map)))

        tf_pose=size/2-np.array([pose[0],-pose[1]])*self.resolution

        coverage=int(distance*self.resolution*1.1)
        xS, yS= tf_pose-coverage
        xE, yE= tf_pose+coverage

        tmp_map=bin_map[int(xS):int(xE), int(yS):int(yE)]

        size=np.array((len(tmp_map[0]),len(tmp_map)))

        mask_pose=(coverage,coverage)

        mask=self.path_mask(mask_pose, direction, steer, distance*self.resolution, size, forward)

        y,x=np.ogrid[:coverage*2, :coverage*2]
        x=np.array(x,np.float32)
        y=np.array(y,np.float32)
        x+=xS-coverage
        y+=yS-coverage
        
        dist=np.sqrt((x-self.__dest[0])**2+(y-self.__dest[1])**2)[mask]
        dist=np.min(dist) if len(dist)>0 else float('inf')
        
        ret=np.sum(tmp_map * mask)
        return (ret, dist, np.ones(mask.shape)*(tmp_map|mask)*255)