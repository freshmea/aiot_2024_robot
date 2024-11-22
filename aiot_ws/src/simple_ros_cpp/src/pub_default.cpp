#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include <chrono>
#include <iostream>

using namespace std;
using namespace std::chrono_literals;

class HellowPublisher : public rclcpp::Node
{
public:
    explicit HellowPublisher()
        : Node("hello"), _count(0)
    {
        _pub = create_publisher<std_msgs::msg::String>("message", 10);
        _timer = create_wall_timer(1s, std::bind(&HellowPublisher::printHello, this));
    }

private:
    int _count;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr _pub;
    rclcpp::TimerBase::SharedPtr _timer;
    void printHello()
    {
        auto msg = std_msgs::msg::String();
        msg.data = "Hello, World!!!!! " + to_string(_count);
        _pub->publish(msg);
        RCLCPP_INFO(get_logger(), msg.data.c_str());
        _count++;
    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<HellowPublisher>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
