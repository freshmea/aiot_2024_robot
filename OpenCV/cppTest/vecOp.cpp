#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    Vec3b p1, p2(0, 0, 255);
    p1[0] = 255;
    cout << "p1: " << p1 << endl;
    cout << "p2: " << p2 << endl;

    Scalar gray = 128;
    Scalar gray2(128, 128, 128);
    cout << "gray: " << gray << endl;
    Scalar red(0, 0, 255);
    cout << "red: " << red << endl;

    Mat img1(400, 400, CV_8UC3, red);
    Mat img2(400, 400, CV_8UC1, gray);
    Mat img3(400, 400, CV_8UC3, gray2);

    Mat img4(400, 400, CV_8UC3, p2);
    Mat img5(400, 400, CV_8UC3, (255, 0, 0)); // black

    imshow("img1", img1);
    imshow("img2", img2);
    imshow("img3", img3);
    imshow("img4", img4);
    imshow("img5", img5);
    waitKey(0);
    return 0;
}
