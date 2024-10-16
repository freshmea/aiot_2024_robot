#include "opencv2/aruco.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

String folder = "/home/aa/aiot_2024_robot/OpenCV/cppTest/data/";
Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);

int main()
{
    Mat src = imread(folder + "markers.jpg");

    vector<int> markerIds;
    vector<vector<Point2f>> markerCorners;
    aruco::detectMarkers(src, dictionary, markerCorners, markerIds);
    cout << markerIds.size() << endl;
    aruco::drawDetectedMarkers(src, markerCorners, markerIds);

    Mat cameraMatrix = (Mat_<double>(3, 3) << 1000, 0, 640, 0, 1000, 360, 0, 0, 1);
    Mat distCoeffs = (Mat_<double>(5, 1) << 0, 0, 0, 0, 0);

    vector<Vec3d> rvecs, tvecs;
    aruco::estimatePoseSingleMarkers(markerCorners, 0.05, cameraMatrix, distCoeffs, rvecs, tvecs);

    if (!markerIds.empty())
    {
        cout << "ID: " << markerIds[0] << endl;
        cout << "rvec " << rvecs[0] << endl;
        cout << "tvec " << tvecs[0] << endl;
    }
    for (int i = 0; i < markerIds.size(); i++)
        cv::drawFrameAxes(src, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1);
    imshow("src", src);
    waitKey();
    return 0;
}
