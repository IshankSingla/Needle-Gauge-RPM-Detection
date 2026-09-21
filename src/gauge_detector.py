import cv2
import numpy as np


def detect_gauge(image):
    """
    Detect the OUTER gauge circle.

    Uses Canny edges followed by Hough Circle detection.

    The outer gauge ring is preferred over smaller
    internal circular structures.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        1
    )

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    # Slightly blur the edge image before Hough detection.
    edges = cv2.GaussianBlur(
        edges,
        (5, 5),
        1
    )

    height, width = gray.shape

    image_center = np.array(
        [width / 2, height / 2],
        dtype=np.float32
    )

    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=80,
        param1=100,
        param2=35,
        minRadius=320,
        maxRadius=390
    )

    if circles is None:

        raise RuntimeError(
            "Could not detect outer gauge circle."
        )

    circles = np.round(
        circles[0]
    ).astype(int)

    candidates = []

    for x, y, radius in circles:

        center = np.array(
            [x, y],
            dtype=np.float32
        )

        # Distance from image center.
        center_distance = np.linalg.norm(
            center -
            image_center
        )

        # Prefer larger circles.
        radius_score = radius

        # We expect the gauge center to be
        # reasonably near the image center,
        # but don't require an exact location.
        center_penalty = (
            center_distance * 0.5
        )

        score = (
            radius_score -
            center_penalty
        )

        candidates.append(
            (
                score,
                x,
                y,
                radius
            )
        )

    # Highest score = best outer gauge candidate.
    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    _, x, y, radius = candidates[0]

    return (
        int(x),
        int(y),
        int(radius)
    )