#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

String folder = "/home/aa/aiot_2024_robot/OpenCV/cppTest/data/";

int main()
{
    Mat src = imread(folder + "frame.png");
    QRCodeDetector detector;

    vector<Point> points;
    String info = detector.detectAndDecode(src, points);

    if (!info.empty())
    {
        polylines(src, points, true, Scalar(0, 0, 255), 2);
        cout << info << endl;
    }
    imshow("src", src);
    waitKey();

    return 0;
}
