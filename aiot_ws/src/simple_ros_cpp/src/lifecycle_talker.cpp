#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "std_msgs/msg/string.hpp"
#include <chrono>
#include <string>

using namespace std::chrono_literals;

class LifecycleTalker : public rclcpp_lifecycle::LifecycleNode
{
public:
    explicit LifecycleTalker(const std::string &node_name, bool intra_process_comms = false)
        : rclcpp_lifecycle::LifecycleNode(node_name, rclcpp::NodeOptions().use_intra_process_comms(intra_process_comms))
    {
    }
    void publish()
    {
        static size_t count = 0;
        auto msg = std::make_unique<std_msgs::msg::String>();
        msg->data = "Lifecycle HelloWorld #" + std::to_string(++count);

        if (!_pub->is_activated())
        {
            RCLCPP_INFO(get_logger(), "Lifecycle pub is inactivate.");
        }
        else
        {
            RCLCPP_INFO(get_logger(), "Lifecycle pub is activate: %s", msg->data.c_str());
        }
        _pub->publish(std::move(msg));
    }

private:
    std::shared_ptr<rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::String>> _pub;
};