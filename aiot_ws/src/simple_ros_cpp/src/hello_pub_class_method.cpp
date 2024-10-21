#include "simple_ros_cpp/hello_pub_class.hpp"

HellowPublisher::HellowPublisher()
    : Node("hello"), _count(0)
{
    _pub = create_publisher<std_msgs::msg::String>("message", 10);
    _timer = create_wall_timer(1s, std::bind(&HellowPublisher::printHello, this));
}

void HellowPublisher::printHello()
{
    auto msg = std_msgs::msg::String();
    msg.data = "Hello, World!!!!! " + to_string(_count);
    _pub->publish(msg);
    RCLCPP_INFO(get_logger(), msg.data.c_str());
    _count++;
}