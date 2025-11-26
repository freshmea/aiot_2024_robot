import math
from typing import Optional  # [추가] None 허용 타입 힌트

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
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

class LoadMapSlam(Node):
    def __init__(self):
        super().__init__("load_map_slam")

        # QoS 설정
        self.map_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )
        self.sensor_qos_profile = QoSProfile(depth=10)


        self.sub_image = self.create_subscription(
            Image,
            "camera/image_raw",
            self.process_image,
            self.sensor_qos_profile,
        )
        self.sub_odom = self.create_subscription(Odometry, "odom", self.odom_callback, self.sensor_qos_profile)
        self.odom_data: Optional[Odometry] = None
        self.pub_map = self.create_publisher(OccupancyGrid, "map", self.map_qos_profile)

        self.cb = CvBridge()

        # [튜닝 파라미터] Alpha 값 설정
        self.bev_spread_alpha = 2.0
        self.bev_spread_beta = 55.0

        # 지도 오프셋 (로봇 중심 ~ BEV 이미지 하단 거리)
        self.map_offset_m = 0.22

        self.converter = CameraConverter(spread_alpha=self.bev_spread_alpha, spread_beta=self.bev_spread_beta)

        # [전역 지도 설정]
        self.map_resolution = 0.0015
        self.map_size = 3000
        self.map_center_x = 1500
        self.map_center_y = 1500

        # 초기값 -1 (Unknown)
        self.global_map = np.full((self.map_size, self.map_size), -1, dtype=np.int8)

        self.ROAD_WIDTH = 110

    def odom_callback(self, msg):
        self.odom_data = msg

    def create_local_grid(self, bev_img):
        """ BEV 이미지 -> 로컬 Occupancy Grid (-1, 0, 100) """
        hsv = cv2.cvtColor(bev_img, cv2.COLOR_BGR2HSV)

        # 차선 추출
        lower_y = np.array([20, 100, 100])
        upper_y = np.array([30, 255, 255])
        mask_y = cv2.inRange(hsv, lower_y, upper_y)

        lower_w = np.array([0, 0, 200])
        upper_w = np.array([180, 30, 255])
        mask_w = cv2.inRange(hsv, lower_w, upper_w)

        obstacles = cv2.bitwise_or(mask_y, mask_w)

        # 로컬 그리드 초기화 (-1)
        h, w = bev_img.shape[:2]
        local_grid = np.full((h, w), -1, dtype=np.int8)

        # 주행 가능 구역 채우기
        for y in range(h):
            y_indices = np.where(mask_y[y] > 0)[0]
            w_indices = np.where(mask_w[y] > 0)[0]

            start_x, end_x = 0, 0
            has_y, has_w = len(y_indices) > 0, len(w_indices) > 0

            if has_y and has_w:
                if y_indices[-1] < w_indices[0]:
                    start_x, end_x = y_indices[-1], w_indices[0]
            elif has_y:
                start_x, end_x = y_indices[-1], min(w, y_indices[-1] + self.ROAD_WIDTH)
            elif has_w:
                start_x, end_x = max(0, w_indices[0] - self.ROAD_WIDTH), w_indices[0]

            if end_x > start_x:
                local_grid[y, start_x:end_x] = 0  # Free Space

        local_grid[obstacles > 0] = 100  # type: ignore
        return local_grid

    def update_global_map(self, local_grid, tx, ty, yaw):
        """ 로컬 그리드를 전역 지도에 누적 """
        h, w = local_grid.shape

        # 1. 변환 행렬 계산 (Local Image -> Global Map Image)
        # Odom 좌표계와 이미지 좌표계 매핑
        # +tx (Forward) -> Moves Up in Image (-global_y)
        # +ty (Left)    -> Moves Left in Image (-global_x)

        global_x = self.map_center_x - int(ty / self.map_resolution)
        global_y = self.map_center_y - int(tx / self.map_resolution)

        # 각도 계산: odom yaw 값을 그대로 사용
        angle = math.degrees(yaw)

        # 로봇의 이미지 상 위치 계산 (오프셋 적용)
        offset_pixels = int(self.map_offset_m / self.map_resolution)
        robot_img_x = w / 2
        robot_img_y = h + offset_pixels

        # 변환 행렬 생성 (회전 -> 이동)
        M = cv2.getRotationMatrix2D((robot_img_x, robot_img_y), angle, 1.0)

        # 이동 변환 추가
        robot_pt = np.array([robot_img_x, robot_img_y, 1.0])
        rotated_pt = np.dot(M, robot_pt)

        M[0, 2] += global_x - rotated_pt[0]
        M[1, 2] += global_y - rotated_pt[1]

        # 2. 로컬 그리드 워핑
        warped_local = cv2.warpAffine(local_grid, M, (self.map_size, self.map_size),
                                      flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=-1) # type: ignore

        # 3. 전역 지도 업데이트
        mask_valid = warped_local != -1
        self.global_map[mask_valid] = warped_local[mask_valid]

    def process_image(self, msg):
        try:
            frame = self.cb.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError:
            return

        # 1. 로컬 데이터 생성
        bev_frame = self.converter.warp_image(frame)
        local_grid = self.create_local_grid(bev_frame)

        # 2. 로봇 위치 확인 및 전역 지도 업데이트
        # [수정] 이미지의 타임스탬프를 전달하여 동기화
        self.get_logger().info(f"Image timestamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")

        if self.odom_data is not None:
            tx = self.odom_data.pose.pose.position.x
            ty = self.odom_data.pose.pose.position.y
            q = self.odom_data.pose.pose.orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            self.update_global_map(local_grid, tx, ty, yaw)

        # 3. 맵 발행 (전역 지도)
        grid_data_flipped = cv2.flip(self.global_map, 0)

        occ_msg = OccupancyGrid()
        current_time = self.get_clock().now().to_msg()
        occ_msg.header.stamp = current_time
        occ_msg.header.frame_id = "odom"
        occ_msg.info.map_load_time = current_time

        occ_msg.info.resolution = self.map_resolution
        occ_msg.info.width = self.map_size
        occ_msg.info.height = self.map_size

        # 지도 원점 설정 (Bottom-Left 기준)
        origin_x = -(self.map_size * self.map_resolution) / 2.0
        origin_y = (self.map_size * self.map_resolution) / 2.0

        occ_msg.info.origin.position.x = origin_x
        occ_msg.info.origin.position.y = origin_y
        occ_msg.info.origin.position.z = 0.0

        occ_msg.info.origin.orientation.x = 0.0
        occ_msg.info.origin.orientation.y = 0.0
        occ_msg.info.origin.orientation.z = -0.7071068
        occ_msg.info.origin.orientation.w = 0.7071068

        occ_msg.data = grid_data_flipped.flatten().tolist()
        self.pub_map.publish(occ_msg)

        # 디버깅
        cv2.imshow("BEV", bev_frame)

        # 전역 지도 시각화 (축소해서 보여주기)
        debug_map = np.zeros_like(self.global_map, dtype=np.uint8)
        debug_map[self.global_map == 0] = 255
        debug_map[self.global_map == 100] = 0
        debug_map[self.global_map == -1] = 127

        cv2.imshow("Global Map", cv2.resize(debug_map, (500, 500)))
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = LoadMapSlam()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()

if __name__ == "__main__":
    main()
