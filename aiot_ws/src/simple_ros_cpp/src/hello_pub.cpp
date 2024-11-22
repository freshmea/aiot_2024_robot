#include "rclcpp/rclcpp.hpp"
#include <chrono>
#include <iostream>

using namespace std;
using namespace std::chrono_literals;
void printHello();

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    // node = Node("hello")
    auto node = std::make_shared<rclcpp::Node>("hello");
    auto timer = node->create_wall_timer(1s, printHello);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

void printHello()
{
    static int count;
    cout << "Hello, World!!!!! " << count << endl;
    count++;
}
