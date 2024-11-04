#include "lifecycle_msgs/msg/transition_event.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include <chrono>
#include <iostream>
#include <string>

using namespace std;
using namespace std::chrono_literals;

class LifecycleListener : public rclcpp::Node
{
public:
    LifecycleListener(const std::string &node_name, const std::string &pub_node)
        : Node(node_name)
    {
        _sub = create_subscription<std_msgs::msg::String>(
            "lifecycle_chatter",
            10,
            std::bind(&LifecycleListener::sub_callback, this, std::placeholders::_1));
        _event_sub = create_subscription<lifecycle_msgs::msg::TransitionEvent>(
            pub_node + "/transition_event",
            10,
            std::bind(&LifecycleListener::event_sub_callback, this, std::placeholders::_1));
    }

private:
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr _sub;
    rclcpp::Subscription<lifecycle_msgs::msg::TransitionEvent>::SharedPtr _event_sub;
    void sub_callback(const std_msgs::msg::String::ConstSharedPtr msg)
    {
        RCLCPP_INFO(get_logger(), "data_callback : %s", msg->data.c_str());
    }
    void event_sub_callback(const lifecycle_msgs::msg::TransitionEvent::ConstSharedPtr msg)
    {
        RCLCPP_INFO(get_logger(), "start state : %s", msg->start_state.label.c_str());
        RCLCPP_INFO(get_logger(), "goal state : %s", msg->goal_state.label.c_str());
    }
};

int main(int argc, char *argv[])
{
    setvbuf(stdout, NULL, _IONBF, BUFSIZ);
    rclcpp::init(argc, argv);
    auto node = std::make_shared<LifecycleListener>("lc_listener", "lc_talker");
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
