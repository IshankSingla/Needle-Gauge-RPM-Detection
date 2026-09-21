def normalize_angle(angle):
    return angle % 360.0


def calculate_gauge_value(
    needle_angle,
    tick_angles,
    value_per_tick=100
):
    """
    Convert needle angle into a continuous gauge value.

    If the needle is sufficiently close to a physical
    tick, the exact tick value is returned.

    Otherwise, the value is calculated continuously
    between the surrounding ticks.

    No expected image values are hardcoded.
    """

    if len(tick_angles) < 2:
        raise ValueError(
            "At least two tick angles are required."
        )

    # -----------------------------------------
    # Normalize tick angles
    # -----------------------------------------

    ticks = [
        normalize_angle(angle)
        for angle in tick_angles
    ]

    start_angle = ticks[0]

    # -----------------------------------------
    # Unwrap ticks
    # -----------------------------------------

    unwrapped_ticks = []

    for angle in ticks:

        while angle < start_angle:
            angle += 360.0

        unwrapped_ticks.append(
            angle
        )

    # -----------------------------------------
    # Normalize needle
    # -----------------------------------------

    needle = normalize_angle(
        needle_angle
    )

    # -----------------------------------------
    # Handle first tick correctly
    # -----------------------------------------

    first_difference = (
        start_angle -
        needle
    )

    if (
        first_difference >= 0
        and
        first_difference <= 2.0
    ):
        needle = start_angle

    elif needle < start_angle:
        needle += 360.0

    # -----------------------------------------
    # Calculate tick spacing
    # -----------------------------------------

    spacings = []

    for i in range(
        len(unwrapped_ticks) - 1
    ):

        spacing = (
            unwrapped_ticks[i + 1]
            -
            unwrapped_ticks[i]
        )

        if spacing > 0:
            spacings.append(
                spacing
            )

    if not spacings:
        raise ValueError(
            "Could not determine tick spacing."
        )

    # Median is more robust than using
    # only one tick interval.
    spacings.sort()

    middle = len(spacings) // 2

    if len(spacings) % 2 == 0:

        median_spacing = (
            spacings[middle - 1]
            +
            spacings[middle]
        ) / 2.0

    else:

        median_spacing = (
            spacings[middle]
        )

    # -----------------------------------------
    # Snap only when very close to a tick
    # -----------------------------------------

    snap_tolerance = (
        median_spacing * 0.15
    )

    nearest_index = None
    nearest_distance = float("inf")

    for i, tick in enumerate(
        unwrapped_ticks
    ):

        distance = abs(
            needle - tick
        )

        if distance < nearest_distance:

            nearest_distance = distance
            nearest_index = i

    if (
        nearest_index is not None
        and
        nearest_distance <= snap_tolerance
    ):

        exact_value = (
            nearest_index *
            value_per_tick
        )

        return (
            float(exact_value),
            nearest_index,
            0.0
        )

    # -----------------------------------------
    # Check lower and upper boundaries
    # -----------------------------------------

    if needle < unwrapped_ticks[0]:

        difference = (
            unwrapped_ticks[0]
            -
            needle
        )

        if difference <= 2.0:

            return (
                0.0,
                0,
                0.0
            )

        raise ValueError(
            "Needle angle is below "
            "the detected gauge scale."
        )

    if needle > unwrapped_ticks[-1]:

        difference = (
            needle
            -
            unwrapped_ticks[-1]
        )

        if difference <= 2.0:

            return (
                float(
                    (len(unwrapped_ticks) - 1)
                    *
                    value_per_tick
                ),
                len(unwrapped_ticks) - 1,
                1.0
            )

        raise ValueError(
            "Needle angle is above "
            "the detected gauge scale."
        )

    # -----------------------------------------
    # Find surrounding ticks
    # -----------------------------------------

    lower_index = None

    for i in range(
        len(unwrapped_ticks) - 1
    ):

        lower = (
            unwrapped_ticks[i]
        )

        upper = (
            unwrapped_ticks[i + 1]
        )

        if (
            lower <= needle <= upper
        ):

            lower_index = i
            break

    if lower_index is None:

        raise ValueError(
            "Could not locate needle "
            "between gauge ticks."
        )

    # -----------------------------------------
    # Calculate continuous position
    # -----------------------------------------

    lower_angle = (
        unwrapped_ticks[
            lower_index
        ]
    )

    upper_angle = (
        unwrapped_ticks[
            lower_index + 1
        ]
    )

    angular_distance = (
        upper_angle -
        lower_angle
    )

    if angular_distance <= 0:

        raise ValueError(
            "Invalid tick spacing."
        )

    fraction = (
        needle -
        lower_angle
    ) / angular_distance

    fraction = max(
        0.0,
        min(
            1.0,
            fraction
        )
    )

    # -----------------------------------------
    # Continuous value
    # -----------------------------------------

    value = (
        lower_index +
        fraction
    ) * value_per_tick

    value = round(value, -1)       # Report the reading to the nearest 10 RPM

    return (
        float(value),
        lower_index,
        float(fraction)
    )