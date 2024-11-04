#include "lifecycle_msgs/msg/state.hpp"
#include "lifecycle_msgs/msg/transition.hpp"
#include "lifecycle_msgs/srv/change_state.hpp"
#include "lifecycle_msgs/srv/get_state.hpp"

#include "rclcpp/rclcpp.hpp"
#include <chrono>
#include <iostream>
#include <thread>

using namespace std;
using namespace std::chrono_literals;

class LifecycleServiceClient : public rclcpp::Node
{
public:
    LifecycleServiceClient()
        : Node("service_client")
    {
        _client = create_client<lifecycle_msgs::srv::ChangeState>("lc_talker/change_state");

        while (!_client->wait_for_service(1s))
        {
            RCLCPP_INFO(get_logger(), "service not available");
        }
        _send_timer = create_wall_timer(3s, std::bind(&LifecycleServiceClient::send_request, this));
    }

private:
    rclcpp::Client<lifecycle_msgs::srv::ChangeState>::SharedPtr _client;
    rclcpp::TimerBase::SharedPtr _send_timer;
    void send_request()
    {
        std::thread t(&LifecycleServiceClient::thread_send_request, this);
        t.detach();
    }
    void thread_send_request()
    {
        _send_timer->cancel();
        auto _request = std::make_shared<lifecycle_msgs::srv::ChangeState::Request>();
        auto transition = lifecycle_msgs::msg::Transition::TRANSITION_CONFIGURE;
        _request->transition.id = transition;
        auto future = _client->async_send_request(_request,
                                                  std::bind(&LifecycleServiceClient::done_callback,
                                                            this,
                                                            std::placeholders::_1));
        future.wait();
        if (future.get()->success)
        {
            RCLCPP_INFO(get_logger(), "Configure seccessful");
            transition = lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE;
            _request->transition.id = transition;
            auto future = _client->async_send_request(_request,
                                                      std::bind(&LifecycleServiceClient::done_callback,
                                                                this,
                                                                std::placeholders::_1));
            future.wait();
            if (future.get()->success)
            {
                RCLCPP_INFO(get_logger(), "Activate seccessful");
            }
        }
        else
        {
            RCLCPP_INFO(get_logger(), "configure error");
        }
    }

    void done_callback(rclcpp::Client<lifecycle_msgs::srv::ChangeState>::SharedFuture future)
    {
        auto response = future.get();
        RCLCPP_INFO(get_logger(), "%s", response->success ? "true" : "false");
    }
};

int main(int argc, char *argv[])
{
    setvbuf(stdout, NULL, _IONBF, BUFSIZ);
    rclcpp::init(argc, argv);
    auto node = std::make_shared<LifecycleServiceClient>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
