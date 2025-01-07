import rclpy
from rclpy.clock import ClockType, ROSClock
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from rclpy.time_source import TimeSource
from std_msgs.msg import Header


class MultiROSClockNode(Node):
    def __init__(self):
        super().__init__("multi_ros_clock_node")

        # 기본 시간 생성
        self.clock = self.get_clock()

        # 두 개의 ROSClock 생성
        self.ros_clock_1 = ROSClock()
        self.ros_clock_2 = ROSClock()
        # self._time_source = TimeSource(node=self)
        # self._time_source.attach_clock(self.ros_clock_1)
        # self._time_source.attach_clock(self.ros_clock_2)

        # ROSClock 1의 초기 시간 설정
        initial_time_1 = Time(seconds=10)
        self.ros_clock_1.set_ros_time_override(initial_time_1)

        # ROSClock 2의 초기 시간 설정
        initial_time_2 = Time(seconds=50)
        self.ros_clock_2.set_ros_time_override(initial_time_2)

        # 퍼블리셔 생성
        self.pub_clock_1 = self.create_publisher(Header, "ros_clock_1", 10)
        self.pub_clock_2 = self.create_publisher(Header, "ros_clock_2", 10)

        # 타이머 생성
        # self.create_timer(1.0, self.time_update)
        self.create_timer(1.0, self.publish_clock_1, clock=self.ros_clock_1)
        self.create_timer(1.0, self.publish_clock_2, clock=self.ros_clock_2)

        # 경과 시간 관리
        self.start_time = self.get_clock().now()

    def publish_clock_1(self):
        msg = Header()
        msg.frame_id = "ros_clock_1"
        msg.stamp = self.ros_clock_1.now().to_msg()
        self.pub_clock_1.publish(msg)
        self.get_logger().info(f"ROSClock 1: {msg.stamp.sec}s {msg.stamp.nanosec}ns")

    def publish_clock_2(self):
        msg = Header()
        msg.frame_id = "ros_clock_2"
        msg.stamp = self.ros_clock_2.now().to_msg()
        self.pub_clock_2.publish(msg)
        self.get_logger().info(f"ROSClock 2: {msg.stamp.sec}s {msg.stamp.nanosec}ns")
        self.ros_clock_2.sleep_for(Duration(seconds=1))
    
    def time_update(self):
        # ROSClock 1 시간 증가,ROSClock 2 시간 증가
        elapsed = self.get_clock().now() - self.start_time
        new_time = Time(seconds=10.0 + elapsed.nanoseconds / 1e9)  # 10초부터 시작
        self.ros_clock_1.set_ros_time_override(new_time)
        new_time = Time(seconds=50.0 + (elapsed.nanoseconds * 2) / 1e9)  # 50초부터 시작, 2배속
        self.ros_clock_2.set_ros_time_override(new_time)



def main():
    rclpy.init()
    node = MultiROSClockNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
