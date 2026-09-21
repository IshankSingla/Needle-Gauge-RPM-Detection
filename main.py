import cv2
import os

from src.gauge_detector import detect_gauge
from src.needle_detector import detect_needle_line
from src.scale_detector import detect_scale_ticks
from src.rpm_calculator import calculate_gauge_value


# --------------------------------
# Folders
# --------------------------------

INPUT_DIR = "data/needle_gauge"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_image(image_path):
    """
    Process one gauge image.
    """

    print("\n" + "=" * 50)
    print(f"Processing: {image_path}")
    print("=" * 50)

    # --------------------------------
    # Read image
    # --------------------------------

    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not read image: {image_path}")
        return

    # --------------------------------
    # 1. Detect gauge
    # --------------------------------

    cx, cy, radius = detect_gauge(image)

    center = (cx, cy)

    print(
        f"Gauge center: "
        f"({cx}, {cy})"
    )

    print(
        f"Gauge radius: {radius}"
    )

    # --------------------------------
    # 2. Detect needle
    # --------------------------------

    needle_angle, p1, p2 = detect_needle_line(
        image,
        center
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
        end_angle,
        scale_radius
    ) = detect_scale_ticks(
        image,
        center
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

    (
        gauge_value,
        lower_tick,
        fraction
    ) = calculate_gauge_value(
        needle_angle,
        tick_angles
    )

    print(
        f"Lower tick: "
        f"{lower_tick}"
    )

    print(
        f"Position between ticks: "
        f"{fraction * 100:.2f}%"
    )

    print(
        f"Gauge reading: "
        f"{gauge_value:.2f} RPM"
    )

    # --------------------------------
    # 5. Create visualization
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

    # Center point
    cv2.circle(
        result,
        center,
        6,
        (255, 0, 0),
        -1
    )

    # Display angle
    cv2.putText(
        result,
        f"Angle: {needle_angle:.2f} deg",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Display RPM
    cv2.putText(
        result,
        f"RPM: {gauge_value:.2f}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # --------------------------------
    # 6. Save result
    # --------------------------------

    filename = os.path.basename(image_path)

    name, extension = os.path.splitext(
        filename
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{name}_result.jpg"
    )

    cv2.imwrite(
        output_path,
        result
    )

    print(
        f"Result saved to: {output_path}"
    )


def main():

    # --------------------------------
    # Find all images
    # --------------------------------

    image_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    )

    image_files = [
        filename
        for filename in os.listdir(INPUT_DIR)
        if filename.lower().endswith(
            image_extensions
        )
    ]

    # --------------------------------
    # Check if images exist
    # --------------------------------

    if not image_files:

        print(
            f"No images found in: "
            f"{INPUT_DIR}"
        )

        return

    print(
        f"Found {len(image_files)} "
        f"image(s)."
    )

    # --------------------------------
    # Process every image
    # --------------------------------

    for filename in image_files:

        image_path = os.path.join(
            INPUT_DIR,
            filename
        )

        try:

            process_image(
                image_path
            )

        except Exception as error:

            print(
                f"ERROR processing "
                f"{filename}: {error}"
            )

    # --------------------------------
    # Finished
    # --------------------------------

    print("\n" + "=" * 50)

    print(
        f"Finished processing "
        f"{len(image_files)} image(s)."
    )

    print(
        f"Results saved in: "
        f"{OUTPUT_DIR}"
    )

    print("=" * 50)


if __name__ == "__main__":
    main()