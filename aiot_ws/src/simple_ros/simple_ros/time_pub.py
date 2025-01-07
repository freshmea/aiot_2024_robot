import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.time_source import TimeSource
from std_msgs.msg import Header


class Time_pub(Node):
    def __init__(self):
        super().__init__("time_pub")
        self.create_timer(1, self.print_hello)
        self.pub = self.create_publisher(Header, "time", 10)
        # self.clock = self.get_clock()
        # ROS_TIME, STEADY_TIME, SYSTEM_TIME
        self.clock = Clock(clock_type=ClockType.ROS_TIME)
        self._time_source = TimeSource(node=self)
        self._time_source.attach_clock(self.clock)

    def print_hello(self):
        msg = Header()
        msg.frame_id = "time"
        msg.stamp = self.clock.now().to_msg()
        # msg.stamp = self.get_clock().now().to_msg()
        print(f"sec: {msg.stamp.sec}, nano sec : {msg.stamp.nanosec}")
        self.pub.publish(msg)

def main():
    rclpy.init()
    node = Time_pub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == "__main__":
    main()
