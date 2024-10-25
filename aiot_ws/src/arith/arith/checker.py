import argparse
import sys

import rclpy
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.node import Node
from rclpy.task import Future
from user_interface.action import ArithmeticChecker
from user_interface.action._arithmetic_checker import (
    ArithmeticChecker_GetResult_Response,
)


class Checker(Node):
    def __init__(self):
        super().__init__("checker")
        self.action_client: ActionClient = ActionClient(self, ArithmeticChecker, "arithmetic_checker")
        self.future = Future()
        self.get_result_future = Future()

    def send_goal(self, goal_sum: str):
        goal_msg : ArithmeticChecker.Goal = ArithmeticChecker.Goal()
        goal_msg.goal_sum = float(goal_sum)
        wait_count = 0
        while(not self.action_client.wait_for_server(timeout_sec=1)):
            self.get_logger().info("checker server is not available!!")
            if wait_count > 10:
                return
            wait_count += 1
        self.future : Future= self.action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self.future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle : ClientGoalHandle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Goal rejected!!!")
            return
        self.get_logger().info("Goal Accepted!!!")
        self.get_result_future : Future = goal_handle.get_result_async()
        self.get_result_future.add_done_callback(self.get_result_callback)
        self.get_logger().info("end response callback!!!")

    def get_result_callback(self, future):
        result : ArithmeticChecker_GetResult_Response = future.result()
        if result.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f"result: {result.result.all_formula}")
            self.get_logger().info(f"result: {result.result.total_sum}")
        if result.status == GoalStatus.STATUS_ABORTED:
            self.get_logger().info("aborted!!")

    def feedback_callback(self, msg):
        feedback : ArithmeticChecker.Feedback = msg.feedback
        self.get_logger().info( f"Received feedback : {feedback.formula}")

def main(args = None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-g',
        '--goal_total_sum',
        type=int,
        default=500,
        help= "Target goal value of total sum"
    )
    parser.add_argument(
        'argv', nargs=argparse.REMAINDER,
        help='Pass arbitrary arguments to the executable'
    )
    args = parser.parse_args()

    rclpy.init(args=args.argv)
    node = Checker()
    node.send_goal(args.goal_total_sum)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()

if __name__ == "__main__":
    main()
