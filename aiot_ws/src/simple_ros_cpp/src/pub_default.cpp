#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include <chrono>
#include <iostream>

using namespace std;
using namespace std::chrono_literals;

class HellowPublisher : public rclcpp::Node
{
public:
    HellowPublisher();

private:
    int _count;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr _pub;
    rclcpp::TimerBase::SharedPtr _timer;
    void printHello();
};

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

int main()
{
    rclcpp::init(0, nullptr);
    auto node = std::make_shared<HellowPublisher>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
