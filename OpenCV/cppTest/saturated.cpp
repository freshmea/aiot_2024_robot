#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

String folder = "/home/aa/aiot_2024_robot/OpenCV/cppTest/data/";
int main()
{
    Mat src = imread(folder + "lenna.bmp", IMREAD_GRAYSCALE);
    Mat brightSrc, saturateSrc;
    brightSrc = src.clone();
    saturateSrc = src.clone();

    for (auto it = brightSrc.begin<uchar>(); it != brightSrc.end<uchar>(); it++)
    {
        *it = *it + 100;
    }

    for (auto it = saturateSrc.begin<uchar>(); it != saturateSrc.end<uchar>(); it++)
    {
        *it = saturate_cast<uchar>(*it + 100);
    }

    Mat addImage, addImage2;
    addImage = src + 100;
    add(src, 100, addImage2);

    imshow("src", src);
    imshow("brightSrc", brightSrc);
    imshow("saturateSrc", saturateSrc);
    imshow("addImage", addImage);
    imshow("addImage2", addImage2);
    waitKey(0);
    return 0;
}
