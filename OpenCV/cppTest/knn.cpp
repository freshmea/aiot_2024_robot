#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace cv::ml;
using namespace std;
String folder = "/home/aa/aiot_2024_robot/OpenCV/cppTest/data/";

Ptr<KNearest> train_knn();

int main()
{
    Ptr<KNearest> knn = train_knn();

    Mat img = Mat::zeros(400, 400, CV_8U);
    imshow("img", img);

    waitKey();
    return 0;
}

Ptr<KNearest> train_knn()
{
    Mat digits = imread(folder + "digits.png", IMREAD_GRAYSCALE);

    Mat train_images, train_labels;
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
            train_labels.push_back(j / 5);

            idx++;
        }
    }

    Ptr<KNearest> knn = KNearest::create();
    knn->train(train_images, ROW_SAMPLE, train_labels);
    return knn;
}
