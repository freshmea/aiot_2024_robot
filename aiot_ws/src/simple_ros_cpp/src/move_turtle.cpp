#include "geometry_msgs/msg/twist.hpp"
#include "rclcpp/rclcpp.hpp"
#include "turtlesim/msg/color.hpp"
#include "turtlesim/msg/pose.hpp"
#include <chrono>
#include <iostream>

using namespace std;
using namespace std::chrono_literals;

class MoveTurtle : public rclcpp::Node
{
public:
    MoveTurtle()
        : Node("hello_sub")
    {
        auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(10))
                               .reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE)
                               .durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);
        _pose_sub = create_subscription<turtlesim::msg::Pose>(
            "turtle1/pose",
            qos_profile,
            std::bind(&MoveTurtle::pose_sub_callback, this, std::placeholders::_1));
        _color_sub = create_subscription<turtlesim::msg::Color>(
            "turtle1/color",
            qos_profile,
            std::bind(&MoveTurtle::color_sub_callback, this, std::placeholders::_1));
        _pub = create_publisher<geometry_msgs::msg::Twist>("turtle1/cmd_vel", qos_profile);
    }

private:
    int _count;
    turtlesim::msg::Pose _pose;
    turtlesim::msg::Color _color;
    rclcpp::Subscription<turtlesim::msg::Pose>::SharedPtr _pose_sub;
    rclcpp::Subscription<turtlesim::msg::Color>::SharedPtr _color_sub;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr _pub;
    void pose_sub_callback(const turtlesim::msg::Pose::SharedPtr msg)
    {
        _pose = *msg;
    }
    void color_sub_callback(const turtlesim::msg::Pose::SharedPtr msg)
    {
        _color = *msg;
    }
};

int main()
{
    rclcpp::init(0, nullptr);
    auto node = std::make_shared<MoveTurtle>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
