#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "std_msgs/msg/string.hpp"
#include <chrono>
#include <iostream>

using namespace std;
using namespace std::chrono_literals;
using namespace cv;

class CannyCamera : public rclcpp::Node
{
public:
    CannyCamera()
        : Node("canny_camera"), _img(240, 320, CV_8UC3, Scalar(255, 255, 255))
    {
        _sub = create_subscription<sensor_msgs::msg::CompressedImage>(
            "/camera/image/compressed",
            10,
            std::bind(&CannyCamera::sub_callback, this, std::placeholders::_1));
    }

private:
    int _count;
    Mat _img;
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr _sub;
    void sub_callback(const sensor_msgs::msg::CompressedImage::SharedPtr msg)
    {

        imshow("img", _img);
        waitKey(30);
    }
};

int main()
{
    rclcpp::init(0, nullptr);
    auto node = std::make_shared<CannyCamera>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
