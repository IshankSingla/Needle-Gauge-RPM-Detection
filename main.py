import cv2

from src.gauge_detector import detect_gauge

from src.needle_detector import (
    create_needle_mask,
    detect_needle_line
)

from src.scale_detector import (
    detect_scale_ticks
)

from src.rpm_calculator import (
    calculate_gauge_value
)

def main():

    IMAGE_PATH = (
        "data/needle_gauge/"
        "895.590073713866.jpg"
    )

    image = cv2.imread(
        IMAGE_PATH
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image: {IMAGE_PATH}"
        )

    # --------------------------------
    # 1. Detect gauge
    # --------------------------------

    cx, cy, radius = detect_gauge(
        image
    )

    center = (cx, cy)

    print(
        f"Gauge center: "
        f"({cx}, {cy})"
    )

    print(
        f"Gauge radius: {radius}"
    )

    # --------------------------------
    # 2. Detect needle line
    # --------------------------------

    needle_angle, p1, p2 = (
        detect_needle_line(image, center)
    )

    print(
        f"Needle angle: "
        f"{needle_angle:.2f} degrees"
    )

    print(
        f"Needle line: "
        f"{p1} -> {p2}"
    )

    # --------------------------------
    # 3. Detect scale ticks
    # --------------------------------

    (
        tick_angles,
        start_angle,
        end_angle
    ) = detect_scale_ticks(
        image,
        center,
        radius
    )

    print(
        f"Detected ticks: "
        f"{len(tick_angles)}"
    )

    print(
        f"Scale start angle: "
        f"{start_angle:.2f}"
    )

    print(
        f"Scale end angle: "
        f"{end_angle:.2f}"
    )

    # --------------------------------
    # 4. Calculate RPM
    # --------------------------------

    scale_value, rpm = (
        calculate_gauge_value(
            needle_angle,
            start_angle,
            end_angle
        )
    )

    print(
        f"Gauge value: "
        f"{scale_value:.2f}"
    )

    print(
        f"RPM: "
        f"{rpm:.2f}"
    )

    # --------------------------------
    # 5. Visualization
    # --------------------------------

    result = image.copy()

    # Gauge circle
    cv2.circle(
        result,
        center,
        radius,
        (255, 0, 0),
        2
    )

    # Needle line
    cv2.line(
        result,
        p1,
        p2,
        (0, 255, 0),
        4
    )

    # Center
    cv2.circle(
        result,
        center,
        6,
        (255, 0, 0),
        -1
    )

    # Display values
    cv2.putText(
        result,
        f"Angle: {needle_angle:.2f} deg",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        result,
        f"RPM: {rpm:.2f}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Needle Gauge Detection",
        result
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()