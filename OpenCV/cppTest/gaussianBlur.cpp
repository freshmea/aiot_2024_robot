#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

String folder = "/home/aa/aiot_2024_robot/OpenCV/cppTest/data/";

int main()
{
    Mat src = imread(folder + "rose.bmp", IMREAD_GRAYSCALE);
    Mat dst;
    int sigma = 3;
    GaussianBlur(src, dst, Size(0, 0), sigma);
    imshow("src", src);
    imshow("dst", dst);
    waitKey();
    return 0;
}
