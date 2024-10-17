#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace cv::ml;
using namespace cv::dnn;
using namespace std;
String folder = "/home/aa/aiot_2024_robot/OpenCV/cppTest/data/";

void on_mouse(int event, int x, int y, int flags, void *data);

int main()
{
    Net net;
    net = readNet(folder + "mnist_cnn.pb");

    Mat img = Mat::zeros(400, 400, CV_8U);
    namedWindow("img");
    setMouseCallback("img", on_mouse, (void *)&img);

    imshow("img", img);
    waitKey();
    Mat blob = blobFromImage(img, 1 / 255.f, Size(28, 28));
    net.setInput(blob);
    Mat prob = net.forward();

    double maxVal;
    Point maxLoc;
    minMaxLoc(prob, NULL, &maxVal, NULL, &maxLoc);
    int digit = maxLoc.x;

    cout << digit << " (" << maxVal * 100 << "% )" << endl;
    return 0;
}

void on_mouse(int event, int x, int y, int flags, void *data)
{
    Mat img = *(Mat *)data;

    static bool flag = false;
    static Point ptPrev;
    if (event == EVENT_LBUTTONDOWN)
    {
        ptPrev = Point(x, y);
        flag = true;
    }
    else if (event == EVENT_LBUTTONUP)
    {
        ptPrev = Point(-1, -1);
        flag = false;
    }
    else if (event == EVENT_MOUSEMOVE && (flag))
    {
        line(img, ptPrev, Point(x, y), Scalar(255, 255, 255), 20, LINE_AA, 0);
        ptPrev = Point(x, y);
        imshow("img", img);
    }
}