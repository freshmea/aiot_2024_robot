#include "lifecycle_msgs/msg/transition_event.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32.hpp"
#include <chrono>
#include <iostream>
#include <string>
#include <wiringPi.h>

using namespace std;
using namespace std::chrono_literals;

class ServoSub : public rclcpp::Node
{
public:
    ServoSub(const std::string &node_name, const std::string &pub_node)
        : Node(node_name), _servo_pin(13)
    {
        _sub = create_subscription<std_msgs::msg::Float32>(
            "servo_angle",
            10,
            std::bind(&ServoSub::sub_callback, this, std::placeholders::_1));
        _event_sub = create_subscription<lifecycle_msgs::msg::TransitionEvent>(
            pub_node + "/transition_event",
            10,
            std::bind(&ServoSub::event_sub_callback, this, std::placeholders::_1));
        wiringPiSetupGpio();
        pinMode(_servo_pin, PWM_OUTPUT);
        pwmSetMode(PWM_MODE_MS);
        pwmSetRange(1024); // range for pwm
        pwmSetClock(375);  // 50hz
    }

private:
    int _servo_pin = 13;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr _sub;
    rclcpp::Subscription<lifecycle_msgs::msg::TransitionEvent>::SharedPtr _event_sub;
    void sub_callback(const std_msgs::msg::Float32::ConstSharedPtr msg)
    {
        int duty_cycle = 26 + (msg->data * 1024) / 180;
        pwmWrite(_servo_pin, duty_cycle);
        RCLCPP_INFO(get_logger(), "data_callback : %f", msg->data);
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
    auto node = std::make_shared<ServoSub>("servo_litener", "servo_talker");
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
