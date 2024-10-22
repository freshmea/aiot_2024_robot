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
        _server = create_service<std_srvs::srv::SetBool>("setBool", std::bind(&ServiceServer::service_callback, this, std::placeholders::_1, std::placeholders::_2));
    }

private:
    typedef std::shared_ptr<std_srvs::srv::SetBool::Request> BRequest;
    typedef std::shared_ptr<std_srvs::srv::SetBool::Response> BResponse;
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr _server;
    void service_callback(const BRequest request, BResponse response)
    {
        std::string str = request->data ? std::string("true") : std::string("false");
        RCLCPP_INFO(get_logger(), str.c_str());
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
