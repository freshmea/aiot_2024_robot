import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from networkx import eccentricity
from rclpy.node import Node
from sensor_msgs.msg import Image


class ColorLineFollower(Node):
    def __init__(self):
        super().__init__('color_line_follower')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()

    def image_callback(self, msg):
        # ROS 이미지 메시지를 OpenCV 이미지로 변환
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 추적할 색상의 범위 설정 (예: 노란색)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # 마스크 이미지에서 윤곽선 찾기
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # image 에 contour 그리기
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

        if contours:
            # 가장 큰 윤곽선을 선택
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M['m00'] > 0:
                # 윤곽선의 중심 계산
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # 화면의 중심과 라인 중심의 차이 계산
                height, width, _ = frame.shape
                error = cx - width // 2

                # PID 제어를 위한 간단한 P 제어기
                k_p = 0.005
                angular_z = -error * k_p

                # 로봇 명령 생성 및 퍼블리시
                twist = Twist()
                twist.linear.x = 0.1  # 전진 속도
                twist.angular.z = angular_z
                self.publisher.publish(twist)
                # 화면의 중심과 라인 중심에 점 그리기
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.circle(frame, (width // 2, height // 2), 5, (255, 0, 0), -1)
            cv2.waitKey(1)
            cv2.imshow('frame', frame)

def main(args=None):
    rclpy.init(args=args)
    color_line_follower = ColorLineFollower()
    try:
        rclpy.spin(color_line_follower)
    except KeyboardInterrupt:
        color_line_follower.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
