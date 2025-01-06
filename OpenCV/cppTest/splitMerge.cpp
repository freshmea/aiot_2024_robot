#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

String folder = "/home/aa/aiot_2024_robot/OpenCV/cppTest/data/";

int main()
{
    Mat src = imread(folder + "candies.png");
    Mat emptyM(src.rows, src.cols, CV_8UC1, Scalar(0));

    vector<Mat> bgr, b_planes, g_planes, r_planes;
    split(src, bgr);
    b_planes = {bgr[0], emptyM, emptyM};
    g_planes = {emptyM, bgr[1], emptyM};
    r_planes = {emptyM, emptyM, bgr[2]};
    Mat bsrc, gsrc, rsrc;
    merge(b_planes, bsrc);
    merge(g_planes, gsrc);
    merge(r_planes, rsrc);

    imshow("src", src);
    int i = 0;
    for (auto img : bgr)
    {
        imshow("bgr" + to_string(i), img);
        i++;
    }
    imshow("bsrc", bsrc);
    imshow("gsrc", gsrc);
    imshow("rsrc", rsrc);

    waitKey();
    return 0;
}
