import cv2
import numpy as np
import rclpy
import tf2_geometry_msgs
from cv_bridge import CvBridge
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformListener


class CameraToFloorMapper(Node):
    def __init__(self):
        super().__init__('camera_to_floor_mapper')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        self.publisher = self.create_publisher(OccupancyGrid, '/map', 10)
        self.bridge = CvBridge()

        # TF Buffer와 Listener 초기화
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 카메라와 지도 설정
        self.map_width = 100
        self.map_height = 100
        self.resolution = 0.1
        self.origin_x = -5.0
        self.origin_y = -5.0
        self.camera_height = 0.14
        self.camera_tilt = 0.4

        # 카메라 내장 매트릭스
        self.camera_matrix = np.array([
            [320, 0, 160],  # fx, 0, cx
            [0, 320, 120],  # 0, fy, cy
            [0, 0, 1]       # 0, 0, 1
        ], dtype=np.float32)
        self.map_data = -np.ones((self.map_height, self.map_width), dtype=np.int8)

    def image_callback(self, msg):
        # ROS 이미지를 OpenCV 이미지로 변환
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self.map_data = -np.ones((self.map_height, self.map_width), dtype=np.int8)
        # 노란색 경로 마스크 생성
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # OccupancyGrid 데이터 초기화

        # 카메라 -> odom 변환 가져오기
        try:
            transform = self.tf_buffer.lookup_transform(
                'odom',  # 부모 프레임
                'camera_link',  # 자식 프레임
                rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(f"TF Transform Error: {e}")
            return

        # 마스크에서 노란색 픽셀 좌표 가져오기
        points = cv2.findNonZero(mask)
        if points is not None:
            for point in points:
                img_x, img_y = point[0]

                # 이미지 픽셀 좌표를 바닥 좌표로 투사
                world_x, world_y = self.pixel_to_world(img_x, img_y)

                # odom 기준으로 변환
                world_point = tf2_geometry_msgs.PointStamped()
                world_point.header.frame_id = 'camera_link'
                world_point.point.x = world_x
                world_point.point.y = world_y
                world_point.point.z = 0.0

                # odom 좌표로 변환
                odom_point = tf2_geometry_msgs.do_transform_point(world_point, transform)

                # 맵 좌표로 변환
                map_x = int((odom_point.point.x - self.origin_x) / self.resolution)
                map_y = int((odom_point.point.y - self.origin_y) / self.resolution)

                if 0 <= map_x < self.map_width and 0 <= map_y < self.map_height:
                    self.map_data[map_y, map_x] = 0  # 노란색 경로를 free로 설정

        # OccupancyGrid 메시지 생성 및 발행
        occupancy_grid = OccupancyGrid()
        occupancy_grid.header.stamp = self.get_clock().now().to_msg()
        occupancy_grid.header.frame_id = 'map'
        occupancy_grid.info.resolution = self.resolution
        occupancy_grid.info.width = self.map_width
        occupancy_grid.info.height = self.map_height
        occupancy_grid.info.origin.position.x = self.origin_x
        occupancy_grid.info.origin.position.y = self.origin_y
        occupancy_grid.info.origin.orientation.w = 1.0
        occupancy_grid.data = self.map_data.flatten().tolist()
        self.publisher.publish(occupancy_grid)

    def pixel_to_world(self, img_x, img_y):
        # 픽셀 좌표를 정규화
        normalized_coords = np.array([[img_x], [img_y], [1]], dtype=np.float32)
        inv_camera_matrix = np.linalg.inv(self.camera_matrix)

        # 정규화된 좌표를 월드 좌표로 변환
        ray = np.dot(inv_camera_matrix, normalized_coords).flatten()

        # 카메라의 기울기 보정
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(self.camera_tilt), -np.sin(self.camera_tilt)],
            [0, np.sin(self.camera_tilt), np.cos(self.camera_tilt)]
        ])
        ray_world = np.dot(rotation_matrix, ray)

        # 바닥 평면과의 교차점 계산 (z = 0)
        scale = self.camera_height / ray_world[1]
        world_x = ray_world[0] * scale
        world_y = ray_world[2] * scale

        return world_x, world_y

def main(args=None):
    rclpy.init(args=args)
    mapper = CameraToFloorMapper()
    rclpy.spin(mapper)
    mapper.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
