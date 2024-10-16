#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

String folder = "/home/aa/aiot_2024_robot/OpenCV/cppTest/data/";

int main()
{
    Mat src = imread(folder + "candies.png");
    int lowerHue = 40;
    int upperHue = 80;
    imshow("src", src);

    cvtColor(src, src, COLOR_BGR2HSV);

    Mat dst;
    inRange(src, Scalar(lowerHue, 50, 0), Scalar(upperHue, 255, 255), dst);
    imshow("dst", dst);

    waitKey();
    return 0;
}
