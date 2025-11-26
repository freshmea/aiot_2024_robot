import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image


class CameraConverter:
    def __init__(self, spread_alpha=1.0, spread_beta=1.0):
        self.width = 320
        self.height = 240

        # 카메라 내부 파라미터
        # horizontal_fov = 3.183 rad (약 182도)
        hfov = 3.183

        # 초점 거리 계산 - 광각 렌즈를 위한 보정
        # fx = width / (2 * tan(hfov/2))이지만 광각에서는 다른 접근 필요
        fx = self.width / (2.0 * np.tan(hfov / 2.0))
        fy = fx
        cx = self.width / 2.0
        cy = self.height / 2.0

        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        # Gazebo 설정에서 distortionK1~K3, T1~T2가 모두 0.0
        # custom_function (tan 모델)은 별도 처리 필요
        # 일단 왜곡 보정 비활성화 (실제 캘리브레이션 필요)
        self.dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # 왜곡 보정 활성화 여부
        self.enable_undistort = True

        # ROI 설정
        self.src_points = np.float32([
            [100, 140], [220, 140],
            [40,  200], [280, 200]
        ]) # type: ignore

        # dst_points를 alpha 값으로 조절하여 펼침 정도 설정
        center_x = 160
        half_width = 80 * spread_alpha

        self.dst_points = np.float32([
            [center_x - half_width-spread_beta,  0],    # Top-Left
            [center_x + half_width+spread_beta,  0],    # Top-Right
            [center_x - half_width+spread_beta,  240],  # Bottom-Left
            [center_x + half_width-spread_beta,  240]   # Bottom-Right
        ]) # type: ignore

        self.perspective_matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points) # type: ignore

        # 왜곡 보정이 활성화된 경우만 최적화된 카메라 행렬 계산
        if self.enable_undistort:
            self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coeffs, (self.width, self.height), 1, (self.width, self.height)
            )
        else:
            self.new_camera_matrix = self.camera_matrix
            self.roi = None

    def undistort_image(self, frame):
        """광각 렌즈 왜곡 보정 (활성화된 경우만)"""
        if not self.enable_undistort:
            return frame
        undistorted = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix)
        # 상하좌우 반전 수정 (-1: 양쪽 축 반전)
        return cv2.flip(undistorted, -1)

    def warp_image(self, frame):
        if frame is None or frame.shape[0] != self.height or frame.shape[1] != self.width:
            frame = cv2.resize(frame, (self.width, self.height))

        # 1단계: 왜곡 보정 (활성화된 경우만)
        if self.enable_undistort:
            undistorted = self.undistort_image(frame)
        else:
            undistorted = frame

        # 2단계: 투시 변환
        return cv2.warpPerspective(undistorted, self.perspective_matrix, (self.width, self.height))

    def draw_roi(self, frame):
        debug_img = frame.copy()
        pts = self.src_points.astype(np.int32)
        roi_poly = np.array([pts[0], pts[1], pts[3], pts[2]], np.int32) # type: ignore
        cv2.polylines(debug_img, [roi_poly], True, (0, 255, 0), 2)
        return debug_img
class LoadFollower(Node):
    def __init__(self):
        super().__init__("load_follower")
        self.qos_profile = QoSProfile(depth=10)
        self.sub_image = self.create_subscription(
            Image, "camera/image_raw", self.process_image, self.qos_profile
        )
        self.pub_cmd = self.create_publisher(Twist, "cmd_vel", self.qos_profile)
        self.cb = CvBridge()

        # BEV 변환 좌표
        self.src_points = np.float32([[100, 140], [220, 140], [20, 200], [300, 200]]) # type: ignore
        self.dst_points = np.float32([[80, 0], [240, 0], [80, 240], [240, 240]]) # type: ignore

        self.line_lost_count = 0
        self.MAX_BLIND_RUN = 15

        # [설정] 도로 폭 (픽셀 단위, BEV 변환 기준)
        self.ROAD_WIDTH = 110

        # 도로 폭 업데이트를 위한 시간 기록 변수
        self.last_width_update_time = self.get_clock().now().nanoseconds / 1e9

        # [튜닝 파라미터] Alpha 값 설정
        self.bev_spread_alpha = 2.0
        self.bev_spread_beta = 55.0

        # 지도 오프셋 (로봇 중심 ~ BEV 이미지 하단 거리)
        self.map_offset_m = 0.25

        self.converter = CameraConverter(spread_alpha=self.bev_spread_alpha, spread_beta=self.bev_spread_beta)

    def birds_eye_view(self, frame):
        h, w = frame.shape[:2]
        matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points) # type: ignore
        return cv2.warpPerspective(frame, matrix, (w, h))

    def get_contour_center(self, mask, min_x=None):
        """ 마스크에서 가장 큰 영역의 중심점 찾기 (min_x 옵션 추가) """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_cx = None
        max_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])

            # [필터링] 기준 좌표(min_x)보다 왼쪽에 있는 영역은 무시
            # 예: 노란선(왼쪽)보다 더 왼쪽에 있는 흰색 물체(건물 등) 무시
            if min_x is not None and cx < min_x:
                continue

            if area > max_area:
                max_area = area
                best_cx = cx

        if best_cx is not None:
            return best_cx, True

        return None, False

    def get_road_center(self, bev_img):
        """ 왼쪽 노란선, 오른쪽 흰선을 인식하여 도로 중심 찾기 """
        hsv = cv2.cvtColor(bev_img, cv2.COLOR_BGR2HSV)

        # 1. 노란선 (왼쪽) 추출
        lower_y = np.array([20, 100, 100])
        upper_y = np.array([30, 255, 255])
        mask_y = cv2.inRange(hsv, lower_y, upper_y)

        # 2. 흰선 (오른쪽) 추출
        # 흰색은 채도(S)가 낮고 명도(V)가 높음
        lower_w = np.array([0, 0, 200])
        upper_w = np.array([180, 30, 255])
        mask_w = cv2.inRange(hsv, lower_w, upper_w)

        # 디버깅용 화면
        debug_mask = cv2.bitwise_or(mask_y, mask_w)
        cv2.imshow("Road Mask (Y+W)", debug_mask)

        # [수정] 노란선 먼저 찾고, 그 위치를 기준으로 흰선 필터링
        cx_y, found_y = self.get_contour_center(mask_y)

        # 흰선 찾을 때: 노란선이 발견되었다면 그보다 오른쪽에 있는 것만 찾음
        min_x_limit = cx_y if found_y else None
        cx_w, found_w = self.get_contour_center(mask_w, min_x=min_x_limit)

        target_x = None
        found = False

        # [시각화] 픽셀 기반 도로 영역 표시 (파란색 반투명)
        road_layer = np.zeros_like(bev_img)
        h, w = bev_img.shape[:2]

        # 행(Row) 단위로 스캔하여 도로 영역 채우기
        for y in range(h):
            # 해당 행에서 노란색/흰색 픽셀 위치 찾기
            y_indices = np.where(mask_y[y] > 0)[0]
            w_indices = np.where(mask_w[y] > 0)[0]

            start_x = 0
            end_x = 0
            has_y = len(y_indices) > 0
            has_w = len(w_indices) > 0

            if has_y and has_w:
                # 노란색의 가장 오른쪽 끝 ~ 흰색의 가장 왼쪽 끝
                l_edge = y_indices[-1]
                r_edge = w_indices[0]

                if l_edge < r_edge:
                    start_x = l_edge
                    end_x = r_edge
            elif has_y:
                # 노란색만 있음 -> 오른쪽으로 ROAD_WIDTH 만큼
                l_edge = y_indices[-1]
                start_x = l_edge
                end_x = min(w, l_edge + self.ROAD_WIDTH)
            elif has_w:
                # 흰색만 있음 -> 왼쪽으로 ROAD_WIDTH 만큼
                r_edge = w_indices[0]
                start_x = max(0, r_edge - self.ROAD_WIDTH)
                end_x = r_edge
            else:
                continue

            # 파란색 채우기 (BGR)
            if end_x > start_x:
                road_layer[y, start_x:end_x] = (255, 0, 0)

        # [수정] 안전한 반투명 합성 방식
        # 전체 이미지에 대해 addWeighted를 수행한 후 마스크 영역만 복사
        if np.any(road_layer > 0):
            alpha = 0.3
            # 전체 이미지를 어둡게 만들고 파란색 레이어를 더함 (전체 크기 유지)
            blended = cv2.addWeighted(bev_img, 1.0 - alpha, road_layer, alpha, 0)

            # road_layer가 있는 픽셀만 원본에 덮어쓰기
            mask_indices = np.any(road_layer > 0, axis=2)
            bev_img[mask_indices] = blended[mask_indices]

        # 주행 로직 (중심점 계산)
        if found_y and found_w:
            # [수정] 역주행 방지 로직: 노란선은 항상 흰선보다 왼쪽에 있어야 함
            if cx_y < cx_w: # type: ignore
                # 정상: 두 선의 중간 지점이 목표
                target_x = (cx_y + cx_w) // 2 # type: ignore
                found = True

                # [추가] 1초마다 도로 폭 업데이트
                current_time = self.get_clock().now().nanoseconds / 1e9
                if current_time - self.last_width_update_time > 1.0:
                    new_width = cx_w - cx_y # type: ignore
                    # 너무 작은 값(노이즈)이 아닐 때만 업데이트 (예: 50픽셀 이상)
                    if new_width > 50:
                        self.ROAD_WIDTH = int(new_width)
                        self.last_width_update_time = current_time
                        print(f"Updated ROAD_WIDTH: {self.ROAD_WIDTH}")
            else:
                # 역주행 감지: 노란선이 흰선보다 오른쪽에 있음 -> 정지
                print("WARNING: Reverse Driving Detected! (Yellow Right, White Left)")
                found = False

        elif found_y:
            # 왼쪽 노란선만 발견: 노란선에서 도로 폭의 절반만큼 오른쪽으로 이동
            target_x = cx_y + (self.ROAD_WIDTH // 2) # type: ignore
            found = True

        elif found_w:
            # 오른쪽 흰선만 발견: 흰선에서 도로 폭의 절반만큼 왼쪽으로 이동
            target_x = cx_w - (self.ROAD_WIDTH // 2) # type: ignore
            found = True

        return target_x, found

    def process_image(self, msg):
        try:
            frame = self.cb.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError:
            return

        # ROI 및 BEV 변환 확인
        roi_check_img = self.converter.draw_roi(frame)
        cv2.imshow("ROI Check", roi_check_img)

        bev_frame_pre = self.converter.warp_image(frame)
        bev_frame = self.birds_eye_view(bev_frame_pre)

        # 도로 중심 찾기
        center_x, found = self.get_road_center(bev_frame)

        twist = Twist()
        h, w, _ = frame.shape
        target_center = w / 2  # 화면 중앙 (320/2 = 160)

        if found:
            self.line_lost_count = 0
            err = center_x - target_center # type: ignore

            twist.linear.x = 0.07
            twist.angular.z = -0.005 * err

            # 시각화
            cv2.circle(bev_frame, (int(center_x), 120), 5, (0, 255, 0), -1) # 초록색 점 # type: ignore
        else:
            if self.line_lost_count < self.MAX_BLIND_RUN:
                twist.linear.x = 0.07
                twist.angular.z = 0.0
                self.line_lost_count += 1
            else:
                twist.linear.x = 0.0
                twist.angular.z = 0.4
                print("Lost road! Searching...")

        self.pub_cmd.publish(twist)
        cv2.imshow("Original", frame)
        cv2.imshow("BEV", bev_frame)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = LoadFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()

if __name__ == "__main__":
    main()
