#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

String folder = "/home/aa/aiot_2024_robot/OpenCV/cppTest/data/";

void histogramGraph(const Mat &src, Mat &graph);
int main()
{
    Mat src = imread(folder + "lenna.bmp", IMREAD_GRAYSCALE);
    Mat dst;
    equalizeHist(src, dst);

    Mat graphSrc(100, 256, CV_8U, Scalar(255));
    Mat graphDst(100, 256, CV_8U, Scalar(255));

    histogramGraph(src, graphSrc);
    histogramGraph(dst, graphDst);
    imshow("src", src);
    imshow("dst", dst);
    imshow("GSrc", graphSrc);
    imshow("GDst", graphDst);
    waitKey();
    return 0;
}

void histogramGraph(const Mat &src, Mat &graph)
{
    Mat histo;
    int channels[] = {0};
    int dims = 1;
    const int histSize[] = {256};
    float graylevel[] = {0, 256};
    const float *ranges[] = {graylevel};

    calcHist(&src, 1, channels, noArray(), histo, dims, histSize, ranges);
    normalize(histo, histo, 0, 100, NORM_MINMAX);
    for (int i = 0; i < 256; i++)
    {
        line(graph, Point(i, 100), Point(i, 100 - cvRound(histo.at<float>(i))), Scalar(0));
    }
}
