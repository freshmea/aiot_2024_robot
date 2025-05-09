import time

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import SetBool


class ServiceServer(Node):
    def __init__(self):
        super().__init__("service_server")
        self.callback_group = ReentrantCallbackGroup()
        self.bool = False
        self.create_service(
            SetBool,
            "setBool",
            self.set_bool_callback,
            callback_group=self.callback_group,
        )
        self.cnt = 0

    def set_bool_callback(self, request, response):
        self.cnt += 1
        self.get_logger().info(f"{self.cnt}번째 요청 처리")
        self.get_logger().info(f"현재 세팅된 불 값 : {self.bool}")
        self.get_logger().info(f"변경 요청한 불 값 : {request.data}")
        if request.data != self.bool:
            self.bool = not self.bool
            response.success = True
            response.message = f"{self.cnt}번째 요청 {self.bool} 처리 완료"
        else:
            response.success = False
            response.message = f"{self.cnt}번째 요청 {self.bool} 처리 실패"
        time.sleep(5)
        return response


def main():
    rclpy.init()
    node = ServiceServer()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()
