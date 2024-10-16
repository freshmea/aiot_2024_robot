#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

String folder = "/home/aa/aiot_2024_robot/OpenCV/cppTest/data/";

int main()
{
    Mat src = imread(folder + "kids.png");

    CascadeClassifier classifier(folder + "haarcascade_frontalface_default.xml");

    vector<Rect> faces;
    classifier.detectMultiScale(src, faces);

    cout << faces.size() << endl;
    for (auto face : faces)
    {
        rectangle(src, face, Scalar(0, 0, 255), 3);
    }
    imshow("src", src);
    waitKey();
    return 0;
}
