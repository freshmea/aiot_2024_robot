#ifndef COMPOSITION__TALKER_COMPONENT_HPP_
#define COMPOSITION__TALKER_COMPONENT_HPP_

#include "composition/visibility_control.h"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

namespace composition
{

    class Talker : public rclcpp::Node
    {
    public:
        COMPOSITION_PUBLIC
        explicit Talker(const rclcpp::NodeOptions &options);

    protected:
        void on_timer();

    private:
        size_t _count;
        rclcpp::Publisher<std_msgs::msg::String>::SharedPtr _pub;
        rclcpp::TimerBase::SharedPtr _timer;
    };
}

#endif // COMPOSITION__TALKER_COMPONENT_HPP_