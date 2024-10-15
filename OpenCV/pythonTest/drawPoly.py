import cv2
import numpy as np


def main():
    red = (0, 0, 255)
    green = (0, 255, 0)
    img = np.full((400, 540, 3), 255, dtype=np.uint8)
    a = 0
    while True:
        cv2.line(img, (10,10),(100+a, 100+a), red, 3)
        cv2.imshow("img", img)
        if cv2.waitKey(10) == 27:
            break
        a += 1

if __name__ == '__main__':
    main()
