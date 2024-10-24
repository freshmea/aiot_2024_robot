#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "user_interface/action/fibonacci.hpp"
#include <chrono>

using namespace std;
using namespace std::placeholders;
using namespace std::chrono_literals;

class ActionClient : public rclcpp::Node
{
public:
    ActionClient()
        : Node("action_client")
    {
        _action_client = rclcpp_action::create_client<Fibonacci>(this, "fibonacci");
    }

    void send_goal(char step[])
    {
        while (!_action_client->wait_for_action_server(1s))
        {
            RCLCPP_INFO(get_logger(), "server is not available!!!");
        }
        auto goal_msg = Fibonacci::Goal();
        goal_msg.step = stoi(step);
        auto options = rclcpp_action::Client<Fibonacci>::SendGoalOptions();
        options.goal_response_callback = std::bind(&ActionClient::goal_response_callback, this, _1);
        options.feedback_callback = std::bind(&ActionClient::feedback_callback, this, _1, _2);
        options.result_callback = std::bind(&ActionClient::result_callback, this, _1);
        _action_client->async_send_goal(goal_msg, options);
    }

private:
    typedef user_interface::action::Fibonacci Fibonacci;
    typedef rclcpp_action::ClientGoalHandle<Fibonacci> GoalHandleFibonacci;
    rclcpp_action::Client<Fibonacci>::SharedPtr _action_client;
    void goal_response_callback(const GoalHandleFibonacci::SharedPtr &goal_handle)
    {
        if (goal_handle)
        {
            RCLCPP_INFO(get_logger(), "Goal accepted by server!");
        }
    }
    void feedback_callback(GoalHandleFibonacci::SharedPtr, const std::shared_ptr<const Fibonacci::Feedback> feedback)
    {
        RCLCPP_INFO(get_logger(), "feedback :");
        for (auto seq : feedback->temp_seq)
        {
            RCLCPP_INFO(get_logger(), "%ld", seq);
        }
    }
    void result_callback(const GoalHandleFibonacci::WrappedResult &result)
    {
        switch (result.code)
        {
        case rclcpp_action::ResultCode::SUCCEEDED:
            RCLCPP_INFO(get_logger(), "result :");
            for (auto seq : result.result->seq)
            {
                RCLCPP_INFO(get_logger(), "%ld", seq);
            }
            return;
        case rclcpp_action::ResultCode::CANCELED:
            RCLCPP_INFO(get_logger(), "canceled!!");
            return;
        case rclcpp_action::ResultCode::ABORTED:
            RCLCPP_INFO(get_logger(), "aborted!!");
            return;
        case rclcpp_action::ResultCode::UNKNOWN:
            RCLCPP_INFO(get_logger(), "unknown!!");
            return;
        }
    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ActionClient>();
    node->send_goal(argv[1]);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
