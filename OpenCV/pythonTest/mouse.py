import cv2
import numpy as np


def on_mouse(event, x, y, flags, param):
    src = param['src']
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y)
        cv2.circle(src, (x,y), 10, (255,0,0), -1)
        cv2.imshow("img", src)


def main():
    folder = "/home/aa/aiot_2024_robot/OpenCV/cppTest/data/"
    src = cv2.imread(folder+"lenna.bmp", cv2.IMREAD_COLOR)
    
    param = {'src':src}
    cv2.namedWindow("img")
    cv2.setMouseCallback("img", on_mouse, param)
    
    cv2.imshow("img", src)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
