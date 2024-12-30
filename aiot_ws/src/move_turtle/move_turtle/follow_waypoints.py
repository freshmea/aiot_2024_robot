import sys

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import FollowWaypoints
from nav2_msgs.action._follow_waypoints import FollowWaypoints_GetResult_Response
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.node import Node
from rclpy.task import Future


class WaypointsFollower(Node):
    def __init__(self):
        super().__init__("action_client")
        self.action_client: ActionClient = ActionClient(self, FollowWaypoints, "follow_waypoints")
        self.future = Future()
        self.get_result_future = Future()

    def send_goal(self, x: str, y: str):
        goal_msg : FollowWaypoints.Goal = FollowWaypoints.Goal()

        # pose setting.
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = 0.0

        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0

        goal_msg.poses.append(pose) # type: ignore

        while(not self.action_client.wait_for_server(timeout_sec=1)):
            self.get_logger().info("fibonacci server is not available!!")
        self.future : Future= self.action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self.future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future : Future):
        goal_handle : ClientGoalHandle = future.result() # type: ignore
        if not goal_handle.accepted:
            self.get_logger().info("Goal rejected!!!")
            return
        self.get_logger().info("Goal Accepted!!!")
        self.get_result_future : Future = goal_handle.get_result_async()
        self.get_result_future.add_done_callback(self.get_result_callback)
        self.get_logger().info("end response callback!!!")

    def get_result_callback(self, future : Future):
        result : FollowWaypoints_GetResult_Response = future.result() # type: ignore
        if result.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f"result: {result.result.missed_waypoints}")
            # shutdown the node
            rclpy.shutdown()

        if result.status == GoalStatus.STATUS_ABORTED:
            self.get_logger().info("aborted!!")

    def feedback_callback(self, msg):
        feedback : FollowWaypoints.Feedback = msg.feedback
        self.get_logger().info( f"Received feedback : {feedback.current_waypoint}")

def main(args = None):
    rclpy.init(args=args)
    node = WaypointsFollower()
    node.send_goal(sys.argv[1], sys.argv[2])
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()

if __name__ == "__main__":
    main()
