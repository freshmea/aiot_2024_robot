#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace cv::ml;
using namespace std;
String folder = "/home/aa/aiot_2024_robot/OpenCV/cppTest/data/";

Ptr<KNearest> train_knn();
void on_mouse(int event, int x, int y, int flags, void *data);

int main()
{
    Ptr<KNearest> knn = train_knn();

    Mat img = Mat::zeros(400, 400, CV_8U);
    namedWindow("img");
    setMouseCallback("img", on_mouse, (void *)&img);

    imshow("img", img);
    waitKey();
    Mat img_resize, img_float, img_flatten, res;

    resize(img, img_resize, Size(20, 20), 0, 0, INTER_AREA);
    img_resize.convertTo(img_float, CV_32F);
    img_flatten = img_float.reshape(1, 1);

    knn->findNearest(img_flatten, 3, res);
    cout << cvRound(res.at<float>(0, 0)) << endl;
    return 0;
}

Ptr<KNearest> train_knn()
{
    Mat digits = imread(folder + "digits.png", IMREAD_GRAYSCALE);

    Mat train_images(5000, 400, CV_32F);
    Mat train_labels(5000, 1, CV_32S);
    int idx = 0;
    for (int j = 0; j < 50; j++)
    {
        for (int i = 0; i < 100; i++)
        {
            Mat roi, roi_float, roi_flatten;
            roi = digits(Rect(i * 20, j * 20, 20, 20));
            cout << i << j << endl;
            roi.convertTo(roi_float, CV_32F);
            roi_flatten = roi_float.reshape(1, 1);

            // 평탄화된 이미지를 train_images에 복사
            roi_flatten.copyTo(train_images.row(idx));

            // 레이블 할당
            train_labels.at<int>(idx, 0) = j / 5;

            idx++;
        }
    }

    Ptr<KNearest> knn = KNearest::create();
    knn->train(train_images, ROW_SAMPLE, train_labels);
    return knn;
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