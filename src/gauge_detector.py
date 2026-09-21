import cv2
import numpy as np


def detect_gauge(image):
    """
    Detect the circular gauge and return:
        center_x
        center_y
        radius
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reduce image noise
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=120,
        param2=50,
        minRadius=250,
        maxRadius=450
    )

    if circles is None:
        raise RuntimeError("Gauge circle could not be detected.")

    circles = np.round(circles[0]).astype(int)

    # Select the largest detected circle
    x, y, radius = max(circles, key=lambda c: c[2])

    return int(x), int(y), int(radius)