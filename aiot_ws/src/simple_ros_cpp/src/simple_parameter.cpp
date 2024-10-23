#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "rclcpp/rclcpp.hpp"

using namespace std;
using namespace std::chrono_literals;

class SimpleParameter : public rclcpp::Node
{
public:
    SimpleParameter()
        : Node("simple_parameter")
    {
        declare_parameter<std::string>("my_para", "내가 만든 파라미터");
        _my_para = get_parameter("my_para").as_string();
        _on_set_para_handle = add_on_set_parameters_callback(std::bind(&SimpleParameter::parameter_callback, this, std::placeholders::_1));
        _print_timer = create_wall_timer(1s, std::bind(&SimpleParameter::print_callback, this));
    }

private:
    std::string _my_para;
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr _on_set_para_handle;
    rclcpp::TimerBase::SharedPtr _print_timer;
    void print_callback()
    {
        RCLCPP_INFO(get_logger(), _my_para.c_str());
    }
    rcl_interfaces::msg::SetParametersResult parameter_callback(const std::vector<rclcpp::Parameter> &parameters)
    {
        for (auto parameter : parameters)
        {
            if (parameter.get_name() == "my_para")
            {
                _my_para = parameter.as_string();
            }
        }
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        return result;
    }
};

int main()
{
    rclcpp::init(0, nullptr);
    auto node = std::make_shared<SimpleParameter>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
