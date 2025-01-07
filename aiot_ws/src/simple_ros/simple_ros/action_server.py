import time

import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from user_interface.action import Fibonacci


class Action_server(Node):
    def __init__(self):
        super().__init__("action_server")
        self.action_server = ActionServer(self,
                                          Fibonacci,
                                          "fibonacci",
                                          execute_callback=self.execute_callback)

    def execute_callback(self, goal_handle):
        self.get_logger().info(f"{goal_handle.request.step}")
        feedback_msg = Fibonacci.Feedback()
        feedback_msg.temp_seq = [0, 1]
        result = Fibonacci.Result()

        for i in range(1, goal_handle.request.step):
            feedback_msg.temp_seq.append(
                feedback_msg.temp_seq[i]+feedback_msg.temp_seq[i-1])
            goal_handle.publish_feedback(feedback_msg)
            time.sleep(1)

        goal_handle.succeed()
        result.seq = feedback_msg.temp_seq
        return result

def main():
    rclpy.init()
    node = Action_server()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == "__main__":
    main()
