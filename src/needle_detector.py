import cv2
import numpy as np


def create_needle_mask(image):
    """
    Detect strongly colored pixels.

    The needle is bright and highly saturated.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    lower = np.array([0, 100, 80])
    upper = np.array([28, 255, 255])

    mask = cv2.inRange(
        hsv,
        lower,
        upper
    )

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask


def detect_needle_line(image, center):
    """
    Detect the gauge needle using Hough line detection.

    The best candidate is selected based on:
        - line length
        - distance from gauge center
        - radial alignment

    The line is oriented from the gauge center
    toward the outer needle tip.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    lower = np.array([0, 100, 80])
    upper = np.array([28, 255, 255])

    mask = cv2.inRange(
        hsv,
        lower,
        upper
    )

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    lines = cv2.HoughLinesP(
        mask,
        rho=1,
        theta=np.pi / 360,
        threshold=40,
        minLineLength=80,
        maxLineGap=30
    )

    if lines is None:
        raise RuntimeError(
            "Could not detect needle line."
        )

    lines = np.asarray(
        lines
    ).reshape(-1, 4)

    cx, cy = center

    candidates = []

    for x1, y1, x2, y2 in lines:

        x1 = float(x1)
        y1 = float(y1)
        x2 = float(x2)
        y2 = float(y2)

        dx = x2 - x1
        dy = y2 - y1

        length = np.hypot(
            dx,
            dy
        )

        if length < 80:
            continue

        # ---------------------------------------
        # Distance from gauge center to line
        # ---------------------------------------

        line_distance = abs(
            dy * cx
            - dx * cy
            + x2 * y1
            - y2 * x1
        ) / length

        if line_distance > 45:
            continue

        # ---------------------------------------
        # Radial alignment
        # ---------------------------------------

        mx = (
            x1 + x2
        ) / 2

        my = (
            y1 + y2
        ) / 2

        radial_dx = mx - cx
        radial_dy = my - cy

        radial_length = np.hypot(
            radial_dx,
            radial_dy
        )

        if radial_length == 0:
            continue

        radial_dx /= radial_length
        radial_dy /= radial_length

        line_dx = dx / length
        line_dy = dy / length

        alignment = abs(
            line_dx * radial_dx
            +
            line_dy * radial_dy
        )

        if alignment < 0.85:
            continue

        # ---------------------------------------
        # Orient line from center toward tip
        # ---------------------------------------

        distance_1 = np.hypot(
            x1 - cx,
            y1 - cy
        )

        distance_2 = np.hypot(
            x2 - cx,
            y2 - cy
        )

        if distance_1 > distance_2:

            tip = (
                int(x1),
                int(y1)
            )

            base = (
                int(x2),
                int(y2)
            )

        else:

            tip = (
                int(x2),
                int(y2)
            )

            base = (
                int(x1),
                int(y1)
            )

        # ---------------------------------------
        # Calculate angle
        # ---------------------------------------

        tip_dx = tip[0] - cx
        tip_dy = tip[1] - cy

        angle = np.degrees(
            np.arctan2(
                tip_dy,
                tip_dx
            )
        )

        angle %= 360

        # ---------------------------------------
        # Score candidate
        # ---------------------------------------

        score = (
            length
            * alignment
            /
            (1 + line_distance)
        )

        candidates.append(
            (
                score,
                length,
                line_distance,
                alignment,
                angle,
                base,
                tip
            )
        )

    if not candidates:
        raise RuntimeError(
            "No suitable needle line found."
        )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    (
        score,
        length,
        line_distance,
        alignment,
        angle,
        base,
        tip
    ) = candidates[0]

    return (
        float(angle),
        base,
        tip
    )