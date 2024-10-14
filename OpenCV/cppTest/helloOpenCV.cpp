#include "opencv2/opencv.hpp"
#include <iostream>

int main()
{
    cv::Mat img;
    img = cv::imread("lena.bmp");
    cv::imshow("image", img);
    std::cout << "Hello OpenCV" << std::endl;
    cv::waitKey(0);
    return 0;
}
