#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

String folder = "/home/aa/aiot_2024_robot/OpenCV/cppTest/data/";

int main()
{
    Mat src = imread(folder + "building.jpg", IMREAD_GRAYSCALE);
    Mat dst1, dst2;

    Canny(src, dst1, 50, 100);
    vector<Vec2f> lines;
    HoughLines(dst1, lines, 1, CV_PI / 180.0, 250);

    Point pt1, pt2;
    for (auto lineP : lines)
    {
        float rho = lineP[0];
        float theta = lineP[1];
        float x0 = rho * cos(theta), y0 = rho * sin(theta);
        pt1.x = cvRound(x0 - 1000 * (-sin(theta)));
        pt1.y = cvRound(y0 - 1000 * cos(theta));
        pt2.x = cvRound(x0 + 1000 * (-sin(theta)));
        pt2.y = cvRound(y0 + 1000 * cos(theta));
        // cout << lineP << endl;
        line(src, pt1, pt2, Scalar(255), 2);
    }
    imshow("src", src);
    imshow("dst1", dst1);
    waitKey();
    return 0;
}
