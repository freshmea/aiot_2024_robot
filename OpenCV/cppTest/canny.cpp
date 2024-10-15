#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

String folder = "/home/aa/aiot_2024_robot/OpenCV/cppTest/data/";

int main()
{
    Mat src = imread(folder + "lenna.bmp", IMREAD_GRAYSCALE);
    Mat dst1, dst2;

    Canny(src, dst1, 50, 100);
    Canny(src, dst2, 100, 150);
    imshow("src", src);
    imshow("dst1", dst1);
    imshow("dst2", dst2);
    waitKey();
    return 0;
}
