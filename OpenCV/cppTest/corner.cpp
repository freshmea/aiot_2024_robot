#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

String folder = "/home/aa/aiot_2024_robot/OpenCV/cppTest/data/";

int main()
{
    Mat src = imread(folder + "building.jpg", IMREAD_GRAYSCALE);

    // vector<KeyPoint> keypoints;
    // FAST(src, keypoints, 60, true);
    // cout << keypoints.size() << endl;
    // for (auto keypoint : keypoints)
    // {
    //     Point pt(keypoint.pt.x, keypoint.pt.y);
    //     circle(src, pt, 10, Scalar(0), -1);
    // }
    vector<Point2f> corners;
    goodFeaturesToTrack(src, corners, 100, 0.01, 10);
    cout << corners.size() << endl;
    for (auto pta : corners)
    {
        Point pt(pta.x, pta.y);
        circle(src, pt, 10, Scalar(0), -1);
    }

        imshow("src", src);
    waitKey();

    return 0;
}
