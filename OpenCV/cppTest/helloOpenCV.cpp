#include "opencv2/opencv.hpp"
#include <iostream>
#include <string>

int main()
{
    cv::Mat img;
    std::cout << img.empty() << std::endl;
    img = cv::imread("lena.bmp");
    if (img.empty())
    {
        std::cout << "Image is empty" << std::endl;
        return -1;
    }

    std::string windowName = "image";
    cv::String windowName2 = "image2";

    cv::imshow("image", img);
    std::cout << "Hello OpenCV" << std::endl;
    cv::imwrite("lena.jpg", img);
    cv::waitKey(0);
    return 0;
}
