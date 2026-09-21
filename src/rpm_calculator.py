def normalize_angle(angle):
    return angle % 360.0


def unwrap_angle(angle, reference):
    """
    Put angle into the same continuous 360-degree
    range as reference.
    """

    angle = normalize_angle(angle)

    while angle < reference:
        angle += 360.0

    while angle >= reference + 360.0:
        angle -= 360.0

    return angle


def calculate_gauge_value(
    needle_angle,
    tick_angles,
    value_per_tick=100
):
    """
    Convert needle angle into a continuous gauge value.

    tick_angles:
        Detected physical tick angles from the image.

    value_per_tick:
        Value represented by one tick.
        For this gauge: 100 RPM.
    """

    if len(tick_angles) < 2:
        raise ValueError(
            "At least two tick angles are required."
        )

    # -----------------------------------------
    # Normalize and unwrap tick angles
    # -----------------------------------------

    ticks = [
        normalize_angle(angle)
        for angle in tick_angles
    ]

    start_angle = ticks[0]

    unwrapped_ticks = []

    for angle in ticks:

        while angle < start_angle:
            angle += 360.0

        unwrapped_ticks.append(angle)

    # -----------------------------------------
    # Normalize needle angle
    # -----------------------------------------

    needle = normalize_angle(
        needle_angle
    )

    while needle < start_angle:
        needle += 360.0

    # -----------------------------------------
    # Allow tiny numerical error at the
    # beginning and end of the gauge.
    # -----------------------------------------

    tolerance = 2.0

    if needle < unwrapped_ticks[0]:

        difference = (
            unwrapped_ticks[0] -
            needle
        )

        if difference <= tolerance:

            needle = unwrapped_ticks[0]

        else:

            raise ValueError(
                "Needle angle is below "
                "the detected gauge scale."
            )


    if needle > unwrapped_ticks[-1]:

        difference = (
            needle -
            unwrapped_ticks[-1]
        )

        if difference <= tolerance:

            needle = unwrapped_ticks[-1]

        else:

            raise ValueError(
                "Needle angle is above "
                "the detected gauge scale."
            )

    # -----------------------------------------
    # Find the two surrounding ticks
    # -----------------------------------------

    lower_index = None

    for i in range(
        len(unwrapped_ticks) - 1
    ):

        lower = unwrapped_ticks[i]
        upper = unwrapped_ticks[i + 1]

        if (
            lower <= needle <= upper
        ):

            lower_index = i
            break

    # -----------------------------------------
    # Needle exactly at final tick
    # -----------------------------------------

    if lower_index is None:

        if (
            abs(
                needle -
                unwrapped_ticks[-1]
            ) < 2.0
        ):

            return (
                float(
                    (len(unwrapped_ticks) - 1)
                    * value_per_tick
                ),
                len(unwrapped_ticks) - 1,
                1.0
            )

        raise ValueError(
            "Could not locate needle "
            "between gauge ticks."
        )

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

    # -----------------------------------------
    # Calculate position between ticks
    # -----------------------------------------

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

    # Keep numerical noise within range
    fraction = max(
        0.0,
        min(1.0, fraction)
    )

    # -----------------------------------------
    # Continuous gauge value
    # -----------------------------------------

    value = (
        lower_index +
        fraction
    ) * value_per_tick

    return (
        float(value),
        lower_index,
        float(fraction)
    )