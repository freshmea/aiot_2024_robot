#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    Scalar green = Scalar(0, 255, 0);
    Scalar red = Scalar(0, 0, 255);
    Mat img(400, 640, CV_8UC3, Scalar(255, 255, 255));

    int a = 0;
    while (true)
    {
        img = Scalar(255, 255, 255);

        rectangle(img, Rect(50 + a, 50 + a, 100, 100), green, 5);
        circle(img, Point(500 - a, 100), 50, red, -1);
        ellipse(img, Point(100 + a, 300 - a), Size(20, 10), 30, 0, 180, green, 5);
        a++;

        if (a > 400)
            a = 0;
        imshow("img", img);
        if (waitKey(10) == 27)
            break;
    }

    return 0;
}
