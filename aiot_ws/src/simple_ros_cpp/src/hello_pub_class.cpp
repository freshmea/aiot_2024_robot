#include "simple_ros_cpp/hello_pub_class.hpp"

HellowPublisher::HellowPublisher()
    : Node("hello"), _count(0)
{
    _pub = this->create_publisher<std_msgs::msg::String>("message", 10);
    _timer = this->create_wall_timer(1s, std::bind(&HellowPublisher::printHello, this));
}

void HellowPublisher::printHello()
{
    auto msg = std_msgs::msg::String();
    msg.data = "Hello, World!!!!! " + to_string(_count);
    _pub->publish(msg);
    cout << "Hello, World!!!!! " << _count << endl;
    _count++;
}

int main()
{
    rclcpp::init(0, nullptr);
    auto node = std::make_shared<HellowPublisher>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
