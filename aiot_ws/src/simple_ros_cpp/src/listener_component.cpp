#include "composition/visibility_control.h"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include <chrono>

using namespace std::chrono_literals;

namespace composition
{
    class Listener : public rclcpp::Node
    {
    public:
        COMPOSITION_PUBLIC
        explicit Listener(const rclcpp::NodeOptions &options)
            : Node("listener", options)
        {
            auto callback = [this](std_msgs::msg::String::ConstSharedPtr msg) -> void
            {
                RCLCPP_INFO(get_logger(), "I heard : %s", msg->data.c_str());
                std::flush(std::cout);
            };
            _sub = create_subscription<std_msgs::msg::String>("chatter", 10, callback);
        }

    private:
        rclcpp::Subscription<std_msgs::msg::String>::SharedPtr _sub;
    };
}
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(composition::Listener)