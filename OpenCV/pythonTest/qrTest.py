import cv2
import numpy as np

folder = "/home/aa/aiot_2024_robot/OpenCV/cppTest/data/"

src = cv2.imread(folder + "frame.png")

detector = cv2.QRCodeDetector()

info, points, _ = detector.detectAndDecode(src)

if info:
    print(info)
    print(points)
    points = np.int32(points).reshape(-1,2)
    cv2.polylines(src, [points], True, (0,0,255), 2)

cv2.imshow("src", src)
cv2.waitKey()

