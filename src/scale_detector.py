import cv2
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


def estimate_outer_radius(image, center):
    """
    Estimate the outer radius of the gauge by finding
    the strongest circular white ring.
    """

    cx, cy = center

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # -----------------------------------------
    # Detect white pixels
    # -----------------------------------------

    white_mask = (
        (gray > 180) &
        (hsv[:, :, 1] < 100)
    ).astype(np.uint8)

    height, width = gray.shape

    max_radius = int(
        min(height, width) * 0.75
    )

    radii = np.arange(
        200,
        max_radius
    )

    ys, xs = np.indices(
        gray.shape
    )

    distances = np.sqrt(
        (xs - cx) ** 2 +
        (ys - cy) ** 2
    )

    profile = []

    for radius in radii:

        ring = (
            (distances >= radius - 1) &
            (distances <= radius + 1)
        )

        if np.any(ring):

            profile.append(
                white_mask[ring].mean()
            )

        else:

            profile.append(0)

    profile = np.array(profile)

    # -----------------------------------------
    # Smooth radial profile
    # -----------------------------------------

    smoothed = gaussian_filter1d(
        profile,
        sigma=2
    )

    peaks, _ = find_peaks(
        smoothed,
        prominence=0.05,
        distance=10
    )

    # -----------------------------------------
    # Keep strong circular structures
    # -----------------------------------------

    valid_peaks = [
        p
        for p in peaks
        if smoothed[p] > 0.4
    ]

    if not valid_peaks:

        raise RuntimeError(
            "Could not detect outer gauge ring."
        )

    # -----------------------------------------
    # Select outermost strong ring
    # -----------------------------------------

    best_peak = max(
        valid_peaks,
        key=lambda p: radii[p]
    )

    outer_radius = int(
        radii[best_peak]
    )

    return outer_radius


def detect_scale_ticks(image, center):
    """
    Detect the angular positions of the gauge ticks.

    Returns:
        tick_angles
        start_angle
        end_angle
        outer_radius
    """

    cx, cy = center

    # -----------------------------------------
    # Detect outer gauge radius
    # -----------------------------------------

    outer_radius = estimate_outer_radius(
        image,
        center
    )

    # -----------------------------------------
    # Define tick detection region
    # -----------------------------------------

    inner_radius = int(
        outer_radius * 0.84
    )

    tick_outer_radius = int(
        outer_radius * 1.02
    )

    # -----------------------------------------
    # Create white-pixel mask
    # -----------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    white_mask = (
        (gray > 180) &
        (hsv[:, :, 1] < 100)
    ).astype(np.uint8)

    # -----------------------------------------
    # Scan angles around gauge
    # -----------------------------------------

    angles = np.linspace(
        140,
        400,
        2601
    )

    radii = np.arange(
        inner_radius,
        tick_outer_radius
    )

    scores = []

    for angle in angles:

        rad = np.deg2rad(angle)

        xs = np.rint(
            cx +
            radii *
            np.cos(rad)
        ).astype(int)

        ys = np.rint(
            cy +
            radii *
            np.sin(rad)
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

    # -----------------------------------------
    # Smooth angular signal
    # -----------------------------------------

    scores = cv2.GaussianBlur(
        scores.reshape(-1, 1),
        (1, 11),
        0
    ).ravel()

    # -----------------------------------------
    # Detect tick peaks
    # -----------------------------------------

    peaks, _ = find_peaks(
        scores,
        distance=50,
        prominence=0.05
    )

    detected_angles = angles[
        peaks
    ]

    # -----------------------------------------
    # Find continuous tick sequence
    # -----------------------------------------

    runs = []

    if len(detected_angles) > 0:

        start = 0

        for i in range(
            1,
            len(detected_angles)
        ):

            spacing = (
                detected_angles[i]
                -
                detected_angles[i - 1]
            )

            # Adjacent ticks should be
            # approximately 7–8 degrees apart.
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
                detected_angles[
                    start:
                ]
            )

    if not runs:

        raise RuntimeError(
            "Could not detect gauge tick marks."
        )

    # -----------------------------------------
    # Select longest continuous sequence
    # -----------------------------------------

    tick_angles = max(
        runs,
        key=len
    )

    if len(tick_angles) < 10:

        raise RuntimeError(
            "Too few gauge ticks detected."
        )

    # -----------------------------------------
    # Use detected physical tick positions
    # directly.
    #
    # IMPORTANT:
    # No polyfit / artificial uniform spacing.
    # -----------------------------------------

    tick_angles = np.asarray(
        tick_angles,
        dtype=np.float64
    )

    # -----------------------------------------
    # Calculate scale boundaries
    # -----------------------------------------

    start_angle = float(
        tick_angles[0]
    )

    end_angle = float(
        tick_angles[-1]
    )

    # -----------------------------------------
    # Return results
    # -----------------------------------------

    return (
        tick_angles,
        start_angle,
        end_angle,
        outer_radius
    )