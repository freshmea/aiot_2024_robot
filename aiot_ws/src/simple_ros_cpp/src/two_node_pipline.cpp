#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int32.hpp"
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <string>

using namespace std;
using namespace std::chrono_literals;

class Producer : public rclcpp::Node
{
public:
    Producer(const std::string &name, const std::string &output)
        : Node(name, rclcpp::NodeOptions().use_intra_process_comms(true))
    {
        _pub = create_publisher<std_msgs::msg::Int32>(output, 10);
        // std::weak_ptr<std::remove_pointer<decltype(_pub.get())>::type> captured_pub = _pub;
        rclcpp::Publisher<std_msgs::msg::Int32>::WeakPtr captured_pub = _pub;
        auto pub_callback = [captured_pub]()
        {
            auto pub_ptr = captured_pub.lock();
            if (!pub_ptr)
            {
                return;
            }
            auto msg = std_msgs::msg::Int32();
            static int32_t count = 0;
            msg.data = count++;
            printf("Pub msg : %d address : 0x%" PRIXPTR "\n", msg->data, reinterpret_cast<std::uintptr_t>(msg.get()));
            pub_ptr->publish(std::move(msg));
        };
        _timer = create_wall_timer(1s, pub_callback);
    }

private:
    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr _pub;
    rclcpp::TimerBase::SharedPtr _timer;
};

class Consumer : public rclcpp::Node
{
public:
    Consumer(const std::string &name, const std::string &output)
        : Node(name, rclcpp::NodeOptions().use_intra_process_comms(true))
    {
        _sub = create_subscription<std_msgs::msg::Int32>(
            output,
            10,
            [](std_msgs::msg::Int32::UniquePtr msg)
            {
                printf("msg value %d address 0x%" PRIXPTR "\n", msg->data,
                       reinterpret_cast<std::uintptr_t>(msg.get()));
            });
    }

private:
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr _sub;
};

int main(int argc, char *argv[])
{
    setvbuf(stdout, NULL, _IONBF, BUFSIZ);
    rclcpp::init(argc, argv);
    rclcpp::executors::SingleThreadedExecutor executor;
    auto producer = std::make_shared<Producer>("producer", "number");
    auto consumer = std::make_shared<Consumer>("consumer", "number");
    executor.add_node(producer);
    executor.add_node(consumer);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
