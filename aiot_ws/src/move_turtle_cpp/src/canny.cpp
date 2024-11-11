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
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }
        _img = cv_ptr->image;
        cv::Mat canny;
        cv::Canny(_img, canny, 100, 200);
        imshow("img", canny);
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
