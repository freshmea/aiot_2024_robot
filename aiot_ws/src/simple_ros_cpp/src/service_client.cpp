#include "rclcpp/rclcpp.hpp"
#include "std_srvs/srv/set_bool.hpp"
#include <chrono>
#include <iostream>

using namespace std;
using namespace std::chrono_literals;

class ServiceClient : public rclcpp::Node
{
public:
    ServiceClient()
        : Node("service_client")
    {
        _client = create_client<std_srvs::srv::SetBool>("setBool");

        while (!_client->wait_for_service(1s))
        {
            RCLCPP_INFO(get_logger(), "service not available");
        }
        _request = std::make_shared<std_srvs::srv::SetBool::Request>();
        _send_timer = create_wall_timer(3s, std::bind(&ServiceClient::send_request, this));
        _update_timer = create_wall_timer(100ms, std::bind(&ServiceClient::update, this));
        // send_request();
    }

private:
    rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr _client;
    std::shared_ptr<std_srvs::srv::SetBool::Request> _request;
    rclcpp::TimerBase::SharedPtr _send_timer;
    rclcpp::TimerBase::SharedPtr _update_timer;
    bool _bool;
    void send_request()
    {
        // _request <- data
        if (!_request->data)
        {
            _request->data = true;
        }
        else
        {
            _request->data = false;
        }

        auto future = _client->async_send_request(_request,
                                                  std::bind(&ServiceClient::done_callback,
                                                            this,
                                                            std::placeholders::_1));
    }
    void done_callback(rclcpp::Client<std_srvs::srv::SetBool>::SharedFuture future)
    {
        auto response = future.get();
        RCLCPP_INFO(get_logger(), response->message.c_str());
        RCLCPP_INFO(get_logger(), "%s", response->success ? "true" : "false");
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
