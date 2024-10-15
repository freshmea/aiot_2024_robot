#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

String folder = "/home/aa/aiot_2024_robot/OpenCV/cppTest/data/";

int main()
{
    Mat src = imread(folder + "rose.bmp", IMREAD_GRAYSCALE);

    float data[] = {-1, -1, 0, -1, 0, 1, 0, 1, 1};
    Mat emboss(3, 3, CV_32F, data);

    Mat dst;
    vector<Mat> dsts;

    filter2D(src, dst, -1, emboss, Point(-1, -1), 128);

    for (int ksize = 3; ksize <= 19; ksize += 2)
    {
        Mat dst;
        blur(src, dst, Size(ksize, ksize));
        dsts.push_back(dst);
    }
    imshow("src", src);
    imshow("dst", dst);
    int i = 3;
    for (auto dst : dsts)
    {
        imshow("dst" + to_string(i), dst);
        i = i + 2;
    }
    waitKey();
    return 0;
}
