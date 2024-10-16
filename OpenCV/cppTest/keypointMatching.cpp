#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

String folder = "/home/aa/aiot_2024_robot/OpenCV/cppTest/data/";

int main()
{
    Mat src1 = imread(folder + "box.png", IMREAD_GRAYSCALE);
    Mat src2 = imread(folder + "box_in_scene.png", IMREAD_GRAYSCALE);

    Ptr<Feature2D> feature = ORB::create();

    vector<KeyPoint> key1, key2;
    Mat desc1, desc2;
    feature->detectAndCompute(src1, Mat(), key1, desc1);
    feature->detectAndCompute(src2, Mat(), key2, desc2);
    cout << desc1.size() << endl;
    cout << desc2.size() << endl;

    Ptr<DescriptorMatcher> matcher = BFMatcher::create(NORM_HAMMING);
    vector<DMatch> matches;
    matcher->match(desc1, desc2, matches);
    cout << matches.size() << endl;

    sort(matches.begin(), matches.end());
    vector<DMatch> good_matches(matches.begin(), matches.begin() + 50);
    Mat dst;
    drawMatches(src1, key1, src2, key2, good_matches, dst);

    vector<Point2f> pts1, pts2;
    int i = 0;
    for (auto pt : good_matches)
    {
        pts1.push_back(key1[pt.queryIdx].pt);
        pts2.push_back(key2[pt.trainIdx].pt);
    }

    Mat M = findHomography(pts1, pts2, RANSAC);

    vector<Point2f> corner1, corner2;
    corner1.push_back(Point2f(0, 0));
    corner1.push_back(Point2f(src1.cols - 1, 0));
    corner1.push_back(Point2f(src1.cols - 1, src1.rows - 1));
    corner1.push_back(Point2f(0, src1.rows - 1));

    perspectiveTransform(corner1, corner2, M);

    vector<Point> corner2_int;
    for (auto pt : corner2)
    {
        corner2_int.push_back(Point(cvRound(pt.x), cvRound(pt.y)));
    }
    polylines(src2, corner2_int, true, Scalar(255), 5);

    imshow("src1", src1);
    imshow("src2", src2);
    imshow("dst", dst);
    waitKey();

    return 0;
}
