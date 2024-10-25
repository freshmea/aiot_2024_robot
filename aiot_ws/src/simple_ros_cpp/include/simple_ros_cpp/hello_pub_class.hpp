#ifndef __HELLO_PUB_CLASS_HPP__
#define __HELLO_PUB_CLASS_HPP__

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
#endif // __HELLO_PUB_CLASS_HPP__