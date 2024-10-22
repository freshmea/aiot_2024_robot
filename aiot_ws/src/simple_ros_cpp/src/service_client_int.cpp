#include "rclcpp/rclcpp.hpp"
#include "user_interface/srv/add_and_odd.hpp"
#include <chrono>
#include <iostream>
#include <random>

// using namespace std;
using namespace std::chrono_literals;

class ServiceClient : public rclcpp::Node
{
public:
    ServiceClient()
        : Node("service_client")
    {
        _client = create_client<user_interface::srv::AddAndOdd>("addandodd");

        while (!_client->wait_for_service(1s))
        {
            RCLCPP_INFO(get_logger(), "service not available");
        }
        _request = std::make_shared<user_interface::srv::AddAndOdd::Request>();
        _send_timer = create_wall_timer(3s, std::bind(&ServiceClient::send_request, this));
        _update_timer = create_wall_timer(500ms, std::bind(&ServiceClient::update, this));
    }

private:
    rclcpp::Client<user_interface::srv::AddAndOdd>::SharedPtr _client;
    std::shared_ptr<user_interface::srv::AddAndOdd::Request> _request;
    rclcpp::TimerBase::SharedPtr _send_timer;
    rclcpp::TimerBase::SharedPtr _update_timer;
    std::random_device rd;
    bool _bool;
    void send_request()
    {
        static std::mt19937 gen(rd());
        static std::uniform_int_distribution<int> dis(5, 20);
        _request->inta = dis(gen);
        _request->intb = dis(gen);

        auto future = _client->async_send_request(_request,
                                                  std::bind(&ServiceClient::done_callback,
                                                            this,
                                                            std::placeholders::_1));
    }
    void done_callback(rclcpp::Client<user_interface::srv::AddAndOdd>::SharedFuture future)
    {
        auto response = future.get();
        RCLCPP_INFO(get_logger(), "%d", response->stamp.sec);
        RCLCPP_INFO(get_logger(), "%d", response->stamp.nanosec);
        RCLCPP_INFO(get_logger(), "%d", response->sum);
        RCLCPP_INFO(get_logger(), response->odd.c_str());
    }
    void update()
    {
        RCLCPP_INFO(get_logger(), "updating now !!!");
    }
};

int main()
{
    rclcpp::init(0, nullptr);
    auto node = std::make_shared<ServiceClient>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
