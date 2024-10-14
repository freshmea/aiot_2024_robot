#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    Mat img;
    // cv::Point_<int> pt;
    Point pt;
    Point2d pt1;
    pt.x = 1.1;
    pt.y = 2.1;
    cout << pt << endl;
    pt1.x = 1.2;
    pt1.y = 2.4;

    Size sz1;
    sz1.width = 10;
    sz1.height = 20;
    Size sz2(100, 200);
    Size sz3(pt1);
    Size2f sz4(3.14, 5.13);
    cout << sz1 << sz2 << sz3 << sz4 << endl;
    cout << sz1.area() << endl;
    cout << sz4.area() << endl;
    cout << sz1.aspectRatio() << endl;

    Rect rt1;
    rt1.x = 10;
    rt1.y = 10;
    rt1.width = 100;
    rt1.height = 200;

    Rect rt2(20, 20, 40, 50);
    Rect rt3(Point(20, 20), Point(40, 50));
    Rect rt4(pt1, sz1);

    cout << rt1 << rt2 << rt3 << rt4 << endl;

    RotatedRect rr1(pt1, sz1, 30);
    Point2f pts[4];
    rr1.points(pts);
    for (auto p : pts)
    {
        cout << p << endl;
    }
    cout << rr1.boundingRect2f() << endl;

    return 0;
}
