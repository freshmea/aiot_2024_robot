#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

String folder = "/home/aa/aiot_2024_robot/OpenCV/cppTest/data/";
void onMouse(int event, int x, int y, int flags, void *data);

Mat src;
bool flag = false;
Point2f srcPts[4], dstPts[4];

int main()
{
    src = imread(folder + "card.bmp", IMREAD_COLOR);

    namedWindow("src");
    setMouseCallback("src", onMouse);
    imshow("src", src);
    waitKey();
    return 0;
}

void onMouse(int event, int x, int y, int flags, void *data)
{
    static int cnt = 0;
    switch (event)
    {
    case EVENT_LBUTTONDOWN:
        srcPts[cnt++] = Point2f(x, y);
        cout << srcPts << endl;
        circle(src, Point(x, y), 10, Scalar(255, 0, 0), -1);
        imshow("img", src);
        if (cnt == 4)
        {
            dstPts[0] = Point2f(0, 0);
            dstPts[1] = Point2f(200 - 1, 0);
            dstPts[2] = Point2f(200 - 1, 300 - 1);
            dstPts[3] = Point2f(0, 300 - 1);
            Mat M = getPerspectiveTransform(srcPts, dstPts);
            Mat dst;
            warpPerspective(src, dst, M, Size(200, 300));
            imshow("dst", dst);
            cnt = 0;
        }
        break;
    }
}