import cv2
import numpy as np


def create_needle_mask(image):
    """
    Detect strongly colored pixels.

    The needle is bright and highly saturated.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Red / orange / yellow range
    lower = np.array([0, 100, 80])
    upper = np.array([35, 255, 255])

    mask = cv2.inRange(
        hsv,
        lower,
        upper
    )

    # Small closing operation to fill tiny gaps
    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask


def find_needle_angle(
    mask,
    center,
    min_radius=40,
    max_radius=None
):
    """
    Find the needle angle using an angular histogram.

    Each colored pixel is converted into an angle
    relative to the gauge center.

    The needle produces a strong concentration of
    pixels at one particular angle.
    """

    cx, cy = center

    height, width = mask.shape

    if max_radius is None:
        max_radius = min(height, width) // 2

    # Coordinates of colored pixels
    ys, xs = np.where(mask > 0)

    if len(xs) < 50:
        raise RuntimeError(
            "Not enough colored pixels detected."
        )

    # Distance from gauge center
    dx = xs.astype(np.float32) - cx
    dy = ys.astype(np.float32) - cy

    radius = np.sqrt(
        dx * dx +
        dy * dy
    )

    # Keep only pixels inside the gauge
    valid = (
        (radius >= min_radius) &
        (radius <= max_radius)
    )

    dx = dx[valid]
    dy = dy[valid]
    radius = radius[valid]

    if len(radius) < 50:
        raise RuntimeError(
            "Not enough pixels inside gauge region."
        )

    # Calculate angle of every pixel
    angles = np.degrees(
        np.arctan2(dy, dx)
    )

    # Convert -180...180 to 0...360
    angles = np.mod(
        angles,
        360.0
    )

    # ------------------------------------------------
    # Build angular histogram
    # ------------------------------------------------

    bins = 720

    histogram, edges = np.histogram(
        angles,
        bins=bins,
        range=(0, 360),
        weights=radius
    )

    # Circular smoothing
    kernel = np.array(
        [1, 2, 3, 4, 3, 2, 1],
        dtype=np.float32
    )

    kernel /= kernel.sum()

    padded = np.concatenate(
        [
            histogram[-3:],
            histogram,
            histogram[:3]
        ]
    )

    smoothed = np.convolve(
        padded,
        kernel,
        mode="same"
    )[3:-3]

    # Strongest angle
    peak_index = np.argmax(
        smoothed
    )

    angle = (
        edges[peak_index] +
        edges[peak_index + 1]
    ) / 2

    return float(angle)


def find_needle_tip(
    mask,
    center,
    needle_angle,
    angle_tolerance=8,
    min_radius=40
):
    """
    Find the outermost colored pixel that lies close
    to the detected needle direction.
    """

    cx, cy = center

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        raise RuntimeError(
            "No colored pixels found."
        )

    dx = xs.astype(np.float32) - cx
    dy = ys.astype(np.float32) - cy

    radius = np.sqrt(
        dx * dx +
        dy * dy
    )

    # Ignore pixels close to the center
    valid = radius >= min_radius

    xs = xs[valid]
    ys = ys[valid]
    dx = dx[valid]
    dy = dy[valid]
    radius = radius[valid]

    angles = np.degrees(
        np.arctan2(dy, dx)
    )

    angles = np.mod(
        angles,
        360.0
    )

    # Circular angular difference
    difference = np.abs(
        angles - needle_angle
    )

    difference = np.minimum(
        difference,
        360 - difference
    )

    valid = difference <= angle_tolerance

    if not np.any(valid):
        raise RuntimeError(
            "Could not find needle tip."
        )

    # Select the farthest pixel along needle direction
    candidate_indices = np.where(
        valid
    )[0]

    best_index = candidate_indices[
        np.argmax(
            radius[candidate_indices]
        )
    ]

    tip = (
        int(xs[best_index]),
        int(ys[best_index])
    )

    return tip


def calculate_needle_angle(
    center,
    tip
):
    """
    Calculate the final geometric angle from
    gauge center to needle tip.
    """

    cx, cy = center
    tx, ty = tip

    dx = tx - cx
    dy = ty - cy

    angle = np.degrees(
        np.arctan2(dy, dx)
    )

    angle = angle % 360

    return float(angle)

def detect_needle_line(image, center):
    """
    Detect the needle line and determine its actual
    direction from the gauge center toward the needle tip.

    Returns:
        angle
        line start
        line end
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # Keep the color range that is currently working
    lower = np.array([0, 100, 80])
    upper = np.array([28, 255, 255])

    mask = cv2.inRange(
        hsv,
        lower,
        upper
    )

    # Connect small gaps
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

    lines = np.asarray(lines).reshape(-1, 4)

    cx, cy = center

    candidates = []

    for x1, y1, x2, y2 in lines:

        x1 = float(x1)
        y1 = float(y1)
        x2 = float(x2)
        y2 = float(y2)

        dx = x2 - x1
        dy = y2 - y1

        length = np.hypot(dx, dy)

        if length < 80:
            continue

        # Distance of both endpoints from gauge center
        distance_1 = np.hypot(
            x1 - cx,
            y1 - cy
        )

        distance_2 = np.hypot(
            x2 - cx,
            y2 - cy
        )

        # The farther endpoint should be the needle tip
        if distance_1 > distance_2:
            tip_x, tip_y = x1, y1
            base_x, base_y = x2, y2
        else:
            tip_x, tip_y = x2, y2
            base_x, base_y = x1, y1

        # Calculate angle FROM CENTER TO TIP
        tip_dx = tip_x - cx
        tip_dy = tip_y - cy

        angle = np.degrees(
            np.arctan2(
                tip_dy,
                tip_dx
            )
        )

        angle %= 360

        # How close is the line to the center?
        line_vector = np.array(
            [dx, dy],
            dtype=np.float32
        )

        line_length = np.linalg.norm(
            line_vector
        )

        line_vector /= line_length

        center_vector = np.array(
            [cx - base_x, cy - base_y],
            dtype=np.float32
        )

        # Distance from center to the line
        cross = abs(
            np.cross(
                line_vector,
                center_vector
            )
        )

        candidates.append(
            (
                cross,
                length,
                angle,
                (int(base_x), int(base_y)),
                (int(tip_x), int(tip_y))
            )
        )

    if not candidates:
        raise RuntimeError(
            "No suitable needle line found."
        )

    # Prefer lines that:
    # 1. pass close to the gauge center
    # 2. are long
    candidates.sort(
        key=lambda x: (
            x[0],
            -x[1]
        )
    )

    (
        center_distance,
        length,
        angle,
        base,
        tip
    ) = candidates[0]

    return (
        float(angle),
        base,
        tip
    )