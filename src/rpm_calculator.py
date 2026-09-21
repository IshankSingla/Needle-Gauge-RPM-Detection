def normalize_angle(angle):
    """
    Convert angle into the same 0–360 range.
    """

    return angle % 360


def calculate_gauge_value(
    needle_angle,
    start_angle,
    end_angle,
    max_value=30,
    multiplier=100
):
    """
    Convert needle angle into gauge value.

    The gauge scale runs from 0 to max_value.

    For this gauge:
        max_value = 30
        multiplier = 100

    Therefore:
        15 -> 1500 RPM
        30 -> 3000 RPM
    """

    needle_angle = normalize_angle(
        needle_angle
    )

    start_angle = normalize_angle(
        start_angle
    )

    end_angle = normalize_angle(
        end_angle
    )

    # The gauge crosses 360 degrees.
    # Example:
    #
    # start = 155
    # end   = 385
    #
    # A needle at 25 degrees becomes:
    #
    # 25 + 360 = 385

    if needle_angle < start_angle:
        needle_angle += 360

    if end_angle < start_angle:
        end_angle += 360

    # Position inside gauge sweep
    position = (
        needle_angle - start_angle
    ) / (
        end_angle - start_angle
    )

    # Keep within gauge limits
    position = max(
        0.0,
        min(1.0, position)
    )

    # Convert position to scale value
    scale_value = (
        position * max_value
    )

    # Convert scale to RPM
    rpm = (
        scale_value * multiplier
    )

    return (
        float(scale_value),
        float(rpm)
    )