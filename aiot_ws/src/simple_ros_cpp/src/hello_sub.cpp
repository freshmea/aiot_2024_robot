#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include <chrono>
#include <iostream>

using namespace std;
using namespace std::chrono_literals;

class HellowSubscriber : public rclcpp::Node
{
public:
    HellowSubscriber()
        : Node("hello_sub")
    {
        _sub = create_subscription<std_msgs::msg::String>(
            "message",
            10,
            std::bind(&HellowSubscriber::sub_callback, this, std::placeholders::_1));
    }

private:
    int _count;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr _sub;
    void sub_callback(const std_msgs::msg::String::SharedPtr msg)
    {
        // cout << msg->data << endl;
        RCLCPP_INFO(get_logger(), msg->data.c_str());
    }
};

int main()
{
    rclcpp::init(0, nullptr);
    auto node = std::make_shared<HellowSubscriber>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
