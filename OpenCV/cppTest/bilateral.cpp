#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

String folder = "/home/aa/aiot_2024_robot/OpenCV/cppTest/data/";

int main()
{
    Mat src = imread(folder + "lenna.bmp", IMREAD_GRAYSCALE);
    Mat noise(src.size(), CV_32SC1);
    randn(noise, 0, 10);
    Mat ori = src.clone();
    add(src, noise, src, Mat(), CV_8U);

    Mat dst1, dst2;
    int sigma = 3;
    GaussianBlur(src, dst1, Size(0, 0), sigma);
    bilateralFilter(src, dst2, -1, 10, 5);

    imshow("ori", ori);
    imshow("src", src);
    imshow("dst1", dst1);
    imshow("dst2", dst2);
    waitKey();
    return 0;
}
