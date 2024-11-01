#include "composition/listener_component.hpp"

namespace composition
{
    explicit Listener::Listener(const rclcpp::NodeOptions &options)
        : Node("listener", options)
    {
        auto callback = [this](std_msgs::msg::String::ConstSharedPtr msg) -> void
        {
            RCLCPP_INFO(get_logger(), "I heard : %s", msg->data.c_str());
            std::flush(std::cout);
        };
        _sub = create_subscription<std_msgs::msg::String>("chatter", 10, callback);
    }
}
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(composition::Listener)