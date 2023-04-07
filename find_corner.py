import cv2
import numpy as np
from random import randint as rnd

camera = cv2.VideoCapture(0)


def nothing(x):
    pass


cv2.namedWindow("frame")
cv2.createTrackbar("H1", "frame", 0, 359, nothing)
cv2.createTrackbar("H2", "frame", 0, 359, nothing)
cv2.createTrackbar("S1", "frame", 0, 255, nothing)
cv2.createTrackbar("S2", "frame", 0, 255, nothing)
cv2.createTrackbar("V1", "frame", 0, 255, nothing)
cv2.createTrackbar("V2", "frame", 0, 255, nothing)

kernel = np.ones((5, 5), np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX

while camera.isOpened():

    _, frame = camera.read()
    img = frame.copy()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    H1 = int(cv2.getTrackbarPos("H1", "frame") / 2)
    H2 = int(cv2.getTrackbarPos("H2", "frame") / 2)
    S1 = cv2.getTrackbarPos("S1", "frame")
    S2 = cv2.getTrackbarPos("S2", "frame")
    V1 = cv2.getTrackbarPos("V1", "frame")
    V2 = cv2.getTrackbarPos("V2", "frame")

    lower = np.array([H1, S1, V1])
    upper = np.array([H2, S2, V2])

    mask = cv2.inRange(hsv, lower, upper)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    res = cv2.bitwise_and(frame, frame, mask)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > 50000 or area < 200:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        print(x, y, w, h)

        color = (rnd(0, 256), rnd(0, 256), rnd(0, 256))
        cv2.drawContours(img, contours, i, color, 6, cv2.LINE_8,
                         hierarchy, 0)
        text = str((w, h))
        cv2.putText(img, text, (x, y), font, 1, color, 2)

    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("img", img)

    if cv2.waitKey(5) == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()