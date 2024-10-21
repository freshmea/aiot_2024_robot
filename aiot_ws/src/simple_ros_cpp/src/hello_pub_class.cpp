#include "simple_ros_cpp/hello_pub_class.hpp"

int main()
{
    rclcpp::init(0, nullptr);
    auto node = std::make_shared<HellowPublisher>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
