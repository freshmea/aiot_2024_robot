#include "rclcpp/rclcpp.hpp"
#include "std_srvs/srv/set_bool.hpp"
#include <chrono>
#include <iostream>

using namespace std;
using namespace std::chrono_literals;

class ServiceServer : public rclcpp::Node
{
public:
    ServiceServer()
        : Node("service_server")
    {
        _server = create_service<std_srvs::srv::SetBool>("setBool", ServiceServer::service_callback);
    }

private:
    typedef std::shared_ptr<std_srvs::srv::SetBool::Request> BRequest;
    typedef std::shared_ptr<std_srvs::srv::SetBool::Response> BResponse;
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr _server;
    void service_callback(const BRequest request, BResponse response)
    {
        RCLCPP_INFO(get_logger(), "incoming service");
        response->message = "sucess";
        response->success = true;
    }
};

int main()
{
    rclcpp::init(0, nullptr);
    auto node = std::make_shared<ServiceServer>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
