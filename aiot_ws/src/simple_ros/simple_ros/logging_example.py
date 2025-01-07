import rclpy
from rclpy.logging import LoggingSeverity
from rclpy.node import Node
from std_msgs.msg import String


class Logger_usage(Node):
    def __init__(self):
        super().__init__("logger_usage_demo")
        self.pub = self.create_publisher(String, "send", 10)
        self.create_timer(1, self.on_timer)
        self.count = 0

    def on_timer(self):
        self.get_logger().log(
            "timer callback called (print once)",
            LoggingSeverity.INFO,
            once=True
        )
        msg = String()
        msg.data = f"Current count : {self.count}"
        self.get_logger().info(f"Publishing: {msg.data}")
        self.pub.publish(msg)

        #function
        if self.debug_function_to_evaluate():
            self.get_logger().debug("Count divides into 12")

        #expression
        if not (self.count % 2):
            self.get_logger().debug("Count is even !!")

        self.count += 1

        if self.count > 15:
            self.get_logger().warn("Resetting count to 0")
            self.count = 0


    def debug_function_to_evaluate(self):
        return is_divide_of_twelve(self.count, self.get_logger())

def is_divide_of_twelve(val, logger):
    if val == 0:
        logger.error("Module divisor cannot be 0")
        return False
    return (12 % val) == 0

def main():
    rclpy.init()
    node = Logger_usage()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == "__main__":
    main()
