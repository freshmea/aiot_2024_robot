#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include <chrono>
#include <iostream>

using namespace std;
using namespace std::chrono_literals;
void printHello();

int main()
{
    rclcpp::init(0, nullptr);
    // node = Node("hello")
    int count = 0;
    auto node = std::make_shared<rclcpp::Node>("hello");
    auto pub = node->create_publisher<std_msgs::msg::String>("message", 10);
    auto timer = node->create_wall_timer(1s, [&count, pub]()
                                         {
        auto msg = std_msgs::msg::String();
        msg.data = "Hello, World!!!!! " + to_string(count);
        pub->publish(msg);
        cout << "Hello, World!!!!! " << count << endl;
        count++; });
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
