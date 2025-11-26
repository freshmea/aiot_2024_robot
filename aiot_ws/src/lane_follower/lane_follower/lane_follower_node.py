import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image


class CameraConverter:
    def __init__(self):
        # XML 설정에 따른 해상도 (320x240)
        self.width = 320
        self.height = 240

        # [핵심] 182도 어안 렌즈의 왜곡을 피하기 위한 ROI 설정
        # 학생들에게 "사다리꼴을 직사각형으로 펴는 것"이라고 설명해주세요.

        # Source Points: 카메라 화면상의 사다리꼴 좌표
        # (좌하, 우하, 좌상, 우상) 순서가 아닌, 변환 함수에 맞춘 순서입니다.
        # 순서: [Top-Left, Top-Right, Bottom-Left, Bottom-Right]
        # Y좌표 120은 화면의 정중앙(지평선 근처)입니다.
        self.src_points = np.float32([
            [100, 140],  # Top-Left  (화면 중앙보다 약간 아래, 왼쪽)
            [220, 140],  # Top-Right (화면 중앙보다 약간 아래, 오른쪽)
            [40,  200],  # Bottom-Left (화면 하단, 왼쪽)
            [280, 200]   # Bottom-Right (화면 하단, 오른쪽)
        ])

        # Destination Points: 지도로 변환했을 때의 직사각형 좌표
        # 실제 비율을 고려하여 정사각형에 가깝게 펼칩니다.
        # 출력 이미지 크기도 320x240을 유지합니다.
        self.dst_points = np.float32([
            [80,  0],    # Top-Left  (지도 상단)
            [240, 0],    # Top-Right
            [80,  240],  # Bottom-Left (지도 하단, 로봇 바로 앞)
            [240, 240]   # Bottom-Right
        ])

        # 변환 행렬(Homography Matrix) 미리 계산
        self.perspective_matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)

        # 역변환 행렬 (지도 좌표 -> 카메라 화면 좌표 변환용, 필요시 사용)
        self.inverse_matrix = cv2.getPerspectiveTransform(self.dst_points, self.src_points)

    def warp_image(self, frame):
        """
        카메라 이미지를 입력받아 Bird's Eye View(지도)로 변환
        """
        # 입력 예외 처리
        if frame is None or frame.shape[0] != self.height or frame.shape[1] != self.width:
            # 해상도가 다를 경우 리사이징 (안전장치)
            frame = cv2.resize(frame, (self.width, self.height))

        # 워핑 수행 (Warp Perspective)
        # flags=cv2.INTER_LINEAR: 부드러운 보간
        bev_img = cv2.warpPerspective(frame, self.perspective_matrix, (self.width, self.height))

        return bev_img

    def draw_roi(self, frame):
        """
        디버깅용: 원본 화면에 우리가 보고 있는 영역(ROI)을 그려서 보여줌
        학생들이 '어디를 보고 지도를 만드는지' 이해하기 좋음
        """
        debug_img = frame.copy()

        # 정수형 좌표로 변환
        pts = self.src_points.astype(np.int32)

        # 사다리꼴 그리기 (순서를 맞춰야 함: TL -> TR -> BR -> BL)
        # src_points 순서가 [TL, TR, BL, BR] 이므로 재정렬 필요
        roi_poly = np.array([pts[0], pts[1], pts[3], pts[2]], np.int32)

        cv2.polylines(debug_img, [roi_poly], True, (0, 255, 0), 2)
        return debug_img


class AdvancedLaneFollower(Node):
    def __init__(self):
        super().__init__("advanced_lane_follower")
        self.qos_profile = QoSProfile(depth=10)

        self.sub_image = self.create_subscription(
            Image, "camera/image_raw", self.process_image, self.qos_profile
        )
        self.pub_cmd = self.create_publisher(Twist, "cmd_vel", self.qos_profile)
        self.cb = CvBridge()

        # [튜닝 포인트 1] BEV 변환을 위한 소스 좌표 (카메라 각도에 따라 조절 필요)
        # 이미지 크기(320x240) 기준 예시
        self.src_points = np.float32([[100, 140], [220, 140], [20, 200], [300, 200]])
        self.dst_points = np.float32([[80, 0], [240, 0], [80, 240], [240, 240]])

        # 교차로 통과를 위한 상태 변수
        self.line_lost_count = 0
        self.MAX_BLIND_RUN = 15  # 선 없이 직진할 횟수 (약 0.5~1초)

        self.converter = CameraConverter()

    def birds_eye_view(self, frame):
        """ 2부 핵심: 원근감을 제거하여 2D 지도로 변환 """
        h, w = frame.shape[:2]

        # 변환 행렬 계산
        matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        # 이미지 워핑 (Top-down View 생성)
        bev_img = cv2.warpPerspective(frame, matrix, (w, h))
        return bev_img

    def get_lane_center(self, bev_img):
        """ 지도(BEV)에서 주행 가능 영역의 중심 찾기 """
        # HSV 변환 및 노란색 추출
        hsv = cv2.cvtColor(bev_img, cv2.COLOR_BGR2HSV)
        lower = np.array([20, 100, 100]) # 노란색 범위
        upper = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        # [지도 데이터화] 디버깅을 위해 마스크 보여주기
        cv2.imshow("Local Map (Binary)", mask)

        # [수정] 전체 픽셀의 무게중심 대신, 연결된 덩어리(Contour) 중 가장 큰 것을 선택
        # 이렇게 하면 노란선이 두 개 이상일 때도 메인 차선을 따라갈 수 있음
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # 가장 큰 영역을 가진 윤곽선 선택 (노이즈 제거 및 메인 차선 선택)
            largest_contour = max(contours, key=cv2.contourArea)

            # 면적이 너무 작으면(노이즈) 무시
            if cv2.contourArea(largest_contour) > 50:
                M = cv2.moments(largest_contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return cx, True

        return None, False

    def process_image(self, msg):
        try:
            frame = self.cb.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            return

        roi_check_img = self.converter.draw_roi(frame)
        cv2.imshow("ROI Check", roi_check_img)

        bev_frame_pre = self.converter.warp_image(frame)
        cv2.imshow("BEV Check", bev_frame_pre)

        # 1. Bird's Eye View 변환 (지도 생성)
        bev_frame = self.birds_eye_view(bev_frame_pre)

        # 2. 지도에서 차선 중심점 찾기
        center_x, found = self.get_lane_center(bev_frame)

        # 3. 주행 로직 (State Machine)
        twist = Twist()
        h, w, _ = frame.shape
        target_center = w / 2  # 화면 중앙

        if found:
            # [상황 A] 차선 감지됨: 정상 주행
            self.line_lost_count = 0
            err = center_x - target_center

            twist.linear.x = 0.07
            twist.angular.z = -0.005 * err  # P-Gain 조절 필요

            # 시각화
            cv2.circle(bev_frame, (center_x, 120), 5, (0, 0, 255), -1)

        else:
            # [상황 B] 차선 없음 (교차로 등)
            if self.line_lost_count < self.MAX_BLIND_RUN:
                # 관성 주행 (직진)
                twist.linear.x = 0.07
                twist.angular.z = 0.0
                self.line_lost_count += 1
                print(f"Blind Run... {self.line_lost_count}")
            else:
                # 진짜 길을 잃음 (탐색 모드)
                twist.linear.x = 0.0
                twist.angular.z = 0.4
                print("Lost line! Searching...")

        self.pub_cmd.publish(twist)

        cv2.imshow("Original", frame)
        cv2.imshow("BEV", bev_frame)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = AdvancedLaneFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()

if __name__ == "__main__":
    main()