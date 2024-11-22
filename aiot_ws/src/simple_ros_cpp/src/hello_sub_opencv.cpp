#include "opencv2/opencv.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include <chrono>
#include <iostream>

using namespace std;
using namespace std::chrono_literals;
using namespace cv;

class HellowSubscriber : public rclcpp::Node
{
public:
    HellowSubscriber()
        : Node("hello_sub"), _img(500, 500, CV_8UC1, Scalar(255))
    {
        _sub = create_subscription<std_msgs::msg::String>(
            "message",
            10,
            std::bind(&HellowSubscriber::sub_callback, this, std::placeholders::_1));
    }

private:
    int _count;
    Mat _img;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr _sub;
    void sub_callback(const std_msgs::msg::String::SharedPtr msg)
    {
        // cout << msg->data << endl;
        RCLCPP_INFO(get_logger(), msg->data.c_str());
        _img = Scalar(255);
        putText(_img, msg->data.c_str(), Point(50, 50), FONT_HERSHEY_PLAIN, 1, Scalar(0), 2);
        imshow("img", _img);
        waitKey(30);
    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<HellowSubscriber>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
