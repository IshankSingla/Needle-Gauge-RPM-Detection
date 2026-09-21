import cv2
import numpy as np


def detect_scale_ticks(image, center, radius):
    """
    Detect the white tick marks around the gauge.

    Returns:
        tick_angles
        start_angle
        end_angle
    """

    cx, cy = center

    # Convert image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # White pixels:
    # high brightness + low saturation
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    white_mask = (
        (gray > 180) &
        (hsv[:, :, 1] < 100)
    ).astype(np.uint8)

    # ------------------------------------------------
    # Analyze the outer part of the gauge
    # ------------------------------------------------

    inner_radius = int(radius * 0.80)
    outer_radius = int(radius * 0.985)

    angles = np.linspace(
        140,
        400,
        2601
    )

    scores = []

    for angle in angles:

        rad = np.deg2rad(angle)

        radii = np.arange(
            inner_radius,
            outer_radius
        )

        xs = np.rint(
            cx + radii * np.cos(rad)
        ).astype(int)

        ys = np.rint(
            cy + radii * np.sin(rad)
        ).astype(int)

        valid = (
            (xs >= 0) &
            (xs < image.shape[1]) &
            (ys >= 0) &
            (ys < image.shape[0])
        )

        values = white_mask[
            ys[valid],
            xs[valid]
        ]

        if len(values) == 0:
            scores.append(0)
        else:
            scores.append(
                values.mean()
            )

    scores = np.array(scores)

    # ------------------------------------------------
    # Smooth signal
    # ------------------------------------------------

    scores = cv2.GaussianBlur(
        scores.reshape(-1, 1),
        (1, 11),
        0
    ).ravel()

    # ------------------------------------------------
    # Find peaks
    # ------------------------------------------------

    from scipy.signal import find_peaks

    peaks, _ = find_peaks(
        scores,
        distance=50,
        prominence=0.05
    )

    detected_angles = angles[peaks]

    # ------------------------------------------------
    # Find the continuous sequence of tick marks
    # ------------------------------------------------

    runs = []

    if len(detected_angles) > 0:

        start = 0

        for i in range(
            1,
            len(detected_angles)
        ):

            spacing = (
                detected_angles[i]
                - detected_angles[i - 1]
            )

            # Tick spacing should be roughly 7–8 degrees
            if not (
                5.5 <= spacing <= 10
            ):

                if i - start >= 5:
                    runs.append(
                        detected_angles[
                            start:i
                        ]
                    )

                start = i

        if (
            len(detected_angles)
            - start >= 5
        ):
            runs.append(
                detected_angles[start:]
            )

    if not runs:
        raise RuntimeError(
            "Could not detect gauge tick marks."
        )

    # Longest continuous sequence
    tick_angles = max(
        runs,
        key=len
    )

    if len(tick_angles) < 10:
        raise RuntimeError(
            "Too few gauge ticks detected."
        )

    start_angle = float(
        tick_angles[0]
    )

    end_angle = float(
        tick_angles[-1]
    )

    return (
        tick_angles,
        start_angle,
        end_angle
    )