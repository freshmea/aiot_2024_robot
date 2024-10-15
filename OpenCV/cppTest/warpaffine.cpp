#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

String folder = "/home/aa/aiot_2024_robot/OpenCV/cppTest/data/";

int main()
{
    Mat src = imread(folder + "tekapo.bmp", IMREAD_GRAYSCALE);
    Mat dst;
    Point2f srcPts[3], dstPts[3];
    srcPts[0] = Point2f(0, 0);
    srcPts[1] = Point2f(src.cols - 1, 0);
    srcPts[2] = Point2f(src.cols - 1, src.rows - 1);
    dstPts[0] = Point2f(src.cols / 2, 0);
    dstPts[1] = Point2f(src.cols - 50, 0);
    dstPts[2] = Point2f(src.cols - 1, src.rows - 1);

    Mat M = getAffineTransform(srcPts, dstPts);
    warpAffine(src, dst, M, Size());

    imshow("src", src);
    imshow("dst", dst);
    waitKey();
    return 0;
}
