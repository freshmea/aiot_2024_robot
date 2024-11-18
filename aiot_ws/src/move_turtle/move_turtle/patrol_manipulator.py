import rclpy
from open_manipulator_msgs.srv import SetJointPosition
from rclpy.clock import Duration
from rclpy.node import Node


class Patrol_manipulator(Node):
    def __init__(self):
        super().__init__("patrol_manipulator")
        self.client = self.create_client(SetJointPosition, "goal_joint_space_path")
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("service not available")
        self.request = SetJointPosition.Request()
        self.create_timer(1/60, self.update)
        self.joint_angles = self.request.joint_position.position
        self.prev_time = self.get_clock().now()
        self.task_done = False

    def send_request(self):
        self.request.joint_position.joint_name = ['joint1', 'joint2', 'joint3', 'joint4']
        self.request.joint_position.position = self.joint_angles
        self.future = self.client.call_async(self.request)
        self.future.add_done_callback(self.done_callback)

    def done_callback(self, future):
        response : SetJointPosition.Response = future.result()
        self.task_done = False
        self.get_logger().info(f"{response.is_planned}")

    def update(self):
        if self.prev_time + Duration(seconds=3) < self.get_clock().now():
            # first move
            self.joint_angles = [0, 0, 0, 0]
            if not self.task_done:
                self.send_request()
                self.task_done = True
        if self.prev_time + Duration(seconds=5) < self.get_clock().now():
            # second move
            self.joint_angles = [0, 0, 0, 0]
            if not self.task_done:
                self.send_request()
                self.task_done = True
        if self.prev_time + Duration(seconds=6) < self.get_clock().now():
            # second move
            self.joint_angles = [0, 0, 0, 0]
            if not self.task_done:
                self.send_request()
                self.task_done = True

        if self.prev_time + Duration(seconds=7) < self.get_clock().now():
            # second move
            self.joint_angles = [0, 0, 0, 0]
            if not self.task_done:
                self.send_request()
                self.task_done = True

        if self.prev_time + Duration(seconds=8) < self.get_clock().now():
            # second move
            self.joint_angles = [0, 0, 0, 0]
            if not self.task_done:
                self.send_request()
                self.task_done = True
        if self.prev_time + Duration(seconds=10) < self.get_clock().now():
            self.prev_time = self.get_clock().now()

def main():
    rclpy.init()
    node = Patrol_manipulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()

if __name__ == "__main__":
    main()
