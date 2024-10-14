#include "opencv2/opencv.hpp"
#include <iostream>

int main()
{
    cv::Mat img;
    // cv::Point_<int> pt;
    cv::Point pt;
    cv::Point2d pt1;
    pt.x = 1.1;
    pt.y = 2.1;
    std::cout << pt << std::endl;
    pt1.x = 1.2;
    pt1.y = 2.4;
    return 0;
}
