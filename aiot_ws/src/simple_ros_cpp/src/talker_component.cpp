#include "composition/talker_component.hpp"
#include <chrono>

using namespace std::chrono_literals;
namespace composition
{
    Talker::Talker(const rclcpp::NodeOptions &options)
        : Node("talker", options), _count(0)
    {
        _pub = create_publisher<std_msgs::msg::String>("chatter", 10);
        _timer = create_wall_timer(1s, std::bind(&Talker::on_timer, this));
    }
    void Talker::on_timer()
    {
        auto msg = std::make_unique<std_msgs::msg::String>();
        msg->data = "Hello, world " + std::to_string(_count);
        _count++;
        RCLCPP_INFO(get_logger(), "Publishing %s", msg->data.c_str());
        _pub->publish(std::move(msg));
    }
}
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(composition::Talker)