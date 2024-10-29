#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int32.hpp"
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <string>

using namespace std;
using namespace std::chrono_literals;

class IncrementerPipe : public rclcpp::Node
{
public:
    IncrementerPipe(const std::string &name, const std::string &in, const std::string &out)
        : Node(name, rclcpp::NodeOptions().use_intra_process_comms(true))
    {
        _pub = create_publisher<std_msgs::msg::Int32>(out, 10);
        rclcpp::Publisher<std_msgs::msg::Int32>::WeakPtr captured_pub = _pub;
        _sub = create_subscription<std_msgs::msg::Int32>(
            in,
            10,
            [this, captured_pub](std_msgs::msg::Int32::UniquePtr msg)
            {
                auto pub_ptr = captured_pub.lock();
                if (!pub_ptr)
                {
                    return;
                }
                // printf("Receive msg value %d address 0x%" PRIXPTR "\n", msg->data,
                //        reinterpret_cast<std::uintptr_t>(msg.get()));
                // printf("sleeping for 1 second\n");
                RCLCPP_INFO(get_logger(), "Receive msg value %d address 0x%" PRIXPTR, msg->data,
                            reinterpret_cast<std::uintptr_t>(msg.get()));
                RCLCPP_INFO(get_logger(), "sleeping for 1 second");
                if (!rclcpp::sleep_for(1s))
                {
                    return;
                }
                // printf("done!\n");
                RCLCPP_INFO(get_logger(), "done!");
                msg->data++;
                // printf("Sending msg value %d address 0x%" PRIXPTR "\n", msg->data,
                //        reinterpret_cast<std::uintptr_t>(msg.get()));
                RCLCPP_INFO(get_logger(), "Sending msg value %d address 0x%" PRIXPTR, msg->data,
                            reinterpret_cast<std::uintptr_t>(msg.get()));
                pub_ptr->publish(std::move(msg));
            });
    }
    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr _pub;
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr _sub;
};

int main(int argc, char *argv[])
{
    setvbuf(stdout, NULL, _IONBF, BUFSIZ);
    rclcpp::init(argc, argv);
    rclcpp::executors::SingleThreadedExecutor executor;
    auto pipe1 = std::make_shared<IncrementerPipe>("pipe1", "topic1", "topic2");
    auto pipe2 = std::make_shared<IncrementerPipe>("pipe2", "topic2", "topic1");
    rclcpp::sleep_for(1s);
    std::unique_ptr<std_msgs::msg::Int32> msg(new std_msgs::msg::Int32());
    msg->data = 42;
    printf("first value: %d address 0x%" PRIXPTR "\n",
           msg->data,
           reinterpret_cast<std::uintptr_t>(msg.get()));
    pipe1->_pub->publish(std::move(msg));
    executor.add_node(pipe1);
    executor.add_node(pipe2);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
