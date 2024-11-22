#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include <chrono>
#include <iostream>

using namespace std;
using namespace std::chrono_literals;

class SyncAsyncSubscriber : public rclcpp::Node
{
public:
    SyncAsyncSubscriber()
        : Node("sync_async_sub")
    {
        _sub = create_subscription<std_msgs::msg::String>(
            "sync_topic",
            10,
            std::bind(&SyncAsyncSubscriber::sub_callback, this, std::placeholders::_1));
        _sub = create_subscription<std_msgs::msg::String>(
            "async_topic",
            10,
            std::bind(&SyncAsyncSubscriber::sub_callback, this, std::placeholders::_1));
    }

private:
    int _count;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr _sub;
    void sub_callback(const std_msgs::msg::String::SharedPtr msg)
    {
        RCLCPP_INFO(get_logger(), msg->data.c_str());
    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SyncAsyncSubscriber>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
