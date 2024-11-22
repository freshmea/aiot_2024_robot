#include "simple_ros_cpp/hello_pub_class.hpp"

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<HellowPublisher>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
