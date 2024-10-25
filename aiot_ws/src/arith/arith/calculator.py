import time

import rclpy
from rclpy.action import ActionServer
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from user_interface.action import ArithmeticChecker
from user_interface.msg import ArithmeticArgument
from user_interface.srv import ArithmeticOperator


class Calculator(Node):
    def __init__(self):
        super().__init__("calculator")
        self.argument_a = 0
        self.argument_b = 0
        self.argument_formula = ''
        self.argument_result = 0.0
        self.argument_operator = '+'
        self.reentrant_group = MutuallyExclusiveCallbackGroup()

        self.create_subscription(
            ArithmeticArgument,
            "arithmetic_argument",
            self.sub_callback, 10
            )
        self.service_server = self.create_service(
            ArithmeticOperator,
            "arithmetic_operator",
            self.service_callback,
            callback_group=self.reentrant_group
            )
        self.action_server = ActionServer(
            self,
            ArithmeticChecker,
            'arithmetic_checker',
            self.execute_callback,
            callback_group=self.reentrant_group)

    def sub_callback(self, msg: ArithmeticArgument):
        self.argument_a = msg.argument_a
        self.argument_b = msg.argument_b
        self.get_logger().info(f"Time Stamp : {msg.stamp}")
        # self.get_logger().info(f"Argument_a : {self.argument_a}")
        # self.get_logger().info(f"Argument_b : {self.argument_b}")
        self.update_formula()

    def service_callback(self, request : ArithmeticOperator.Request, response: ArithmeticOperator.Response):

        if request.arithmetic_operator == ArithmeticOperator.Request.PLUS:
            self.argument_operator = '+'
        if request.arithmetic_operator == ArithmeticOperator.Request.MINUS:
            self.argument_operator = '-'
        if request.arithmetic_operator == ArithmeticOperator.Request.MULTIPLY:
            self.argument_operator = 'X'
        if request.arithmetic_operator == ArithmeticOperator.Request.DIVISION:
            self.argument_operator = '/'
        self.update_formula()
        response.arithmetic_result = self.argument_result
        return response

    def execute_callback(self, goal_handle):
        # tt = ArithmeticChecker.Goal()
        self.get_logger().info(f"{goal_handle.request.goal_sum}")
        feedback_msg = ArithmeticChecker.Feedback()
        feedback_msg.formula = []
        result = ArithmeticChecker.Result()
        sum_temp = 0

        while sum_temp < goal_handle.request.goal_sum:
            feedback_msg.formula.append(self.argument_formula)
            goal_handle.publish_feedback(feedback_msg)
            sum_temp += self.argument_result
            time.sleep(1)

        goal_handle.succeed()
        result.total_sum = sum_temp
        result.all_formula = feedback_msg.formula
        return result

    def update_formula(self):
        if self.argument_operator == '+':
            self.argument_result = self.argument_a + self.argument_b
        if self.argument_operator == '-':
            self.argument_result = self.argument_a - self.argument_b
        if self.argument_operator == 'X':
            self.argument_result = self.argument_a * self.argument_b
        if self.argument_operator == '/':
            try:
                self.argument_result = self.argument_a / self.argument_b
            except ZeroDivisionError:
                self.argument_result = 0.0
        self.argument_formula = f"{self.argument_result} = {self.argument_a} {self.argument_operator} {self.argument_b}"
        self.get_logger().info(self.argument_formula)

def main():
    rclpy.init()
    try:
        node = Calculator()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        node.destroy_node()
        executor.shutdown()

if __name__ == "__main__":
    main()
