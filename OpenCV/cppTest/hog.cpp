#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

String folder = "/home/aa/aiot_2024_robot/OpenCV/cppTest/data/";

int main()
{
    // Mat src = imread(folder + "street.jpg");
    VideoCapture cap(folder + "vtest.avi");
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    while (true)
    {
        Mat src;
        cap >> src;
        vector<Rect> detected;
        hog.detectMultiScale(src, detected);
        for (auto d : detected)
        {
            rectangle(src, d, Scalar(0, 0, 255), 3);
        }
        imshow("src", src);
        if (waitKey(100) == 27)
            break;
    }
    return 0;
}
