#include "rclcpp/rclcpp.hpp"
#include "rcutils/cmdline_parser.h"
#include "user_interface/msg/arithmetic_argument.hpp"
#include <cstdio>
#include <random>
#include <utility>

using namespace std::chrono_literals;
class Argument : public rclcpp::Node
{
public:
    explicit Argument(const rclcpp::NodeOptions &node_options = rclcpp::NodeOptions());

private:
    int _count;
    rclcpp::Publisher<user_interface::msg::ArithmeticArgument>::SharedPtr _pub;
    rclcpp::TimerBase::SharedPtr _timer;
    void pub_callback();
};

Argument::Argument(const rclcpp::NodeOptions &node_options)
    : Node("argument", node_options)
{
    _pub = create_publisher<user_interface::msg::ArithmeticArgument>("argument", 10);
    _timer = create_wall_timer(1s, std::bind(&Argument::pub_callback, this));
}

void Argument::pub_callback()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0, 100);
    user_interface::msg::ArithmeticArgument msg;
    msg.stamp = now();
    msg.argument_a = dis(gen);
    msg.argument_b = dis(gen);
    _pub->publish(msg);
}

void print_help()
{
    printf("for argument node: \n");
    printf("node name [-h]\n");
    printf("Option:\n");
    printf("add some explanation!!!\n");
}

int main(int argc, char *argv[])
{
    if (rcutils_cli_option_exist(argv, argv + argc, "-h"))
    {
        print_help();
        return 0;
    }
    rclcpp::init(argc, argv);
    auto node = std::make_shared<Argument>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}