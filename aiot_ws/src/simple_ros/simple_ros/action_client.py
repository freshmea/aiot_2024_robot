import sys

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from user_interface.action import Fibonacci


class Action_client(Node):
    def __init__(self):
        super().__init__("action_client")
        self.action_client = ActionClient(self, Fibonacci, "fibonacci")

    def send_goal(self, step):
        goal_msg = Fibonacci.Goal()
        goal_msg.step = int(step)
        while(not self.action_client.wait_for_server(timeout_sec=1)):
            self.get_logger().info("fibonacci server is not available!!")
        self.future = self.action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self.future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Goal rejected!!!")
            return
        self.get_logger().info("Goal Accepted!!!")
        self.get_result_future = goal_handle.get_result_async()
        self.get_result_future.add_done_callback(self.get_result_callback)
        self.get_logger().info("end response callback!!!")

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f"result: {result.seq}")

    def feedback_callback(self, msg):
        self.get_logger().info( f"Received feedback : {msg.feedback.temp_seq}")

def main(args = None):
    rclpy.init(args=args)
    node = Action_client()
    node.send_goal(sys.argv[1])
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()

if __name__ == "__main__":
    main()
