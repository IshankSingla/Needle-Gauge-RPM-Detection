# Needle Gauge RPM Detection

A Python and OpenCV-based computer vision system that automatically detects the needle position on an analog RPM gauge image and calculates the corresponding RPM value.

The system processes multiple gauge images automatically, without relying on filenames or manually assigned expected values — every RPM reading is computed purely from detected visual information (gauge geometry, tick positions, and needle angle).

---

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Output](#output)
- [Error Handling](#error-handling)
- [Design Principle](#design-principle)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Modules](#modules)

---

## Features

- Automatic gauge detection (center + radius)
- Automatic scale / tick mark detection
- Needle detection via color segmentation + Hough Line detection
- Precise needle angle calculation
- Continuous interpolation between ticks for accurate readings
- RPM output rounded to the nearest 10
- Batch processing of multiple images in one run
- Annotated output images showing all detected features
- Graceful error handling — one bad image won't stop the batch
- No filename-based or hardcoded RPM logic

---

## How It Works

The system follows a computer vision pipeline:

```
Input Gauge Image
        │
        ▼
  Detect Gauge
        │
        ▼
Find Gauge Center & Radius
        │
        ▼
Detect Scale / Tick Marks
        │
        ▼
   Detect Needle
        │
        ▼
Calculate Needle Angle
        │
        ▼
Find Needle Position Between Ticks
        │
        ▼
   Calculate RPM
        │
        ▼
Round Result to Nearest 10 RPM
        │
        ▼
  Save Annotated Image
```

### 1. Gauge Detection
The circular gauge is detected using OpenCV image processing techniques, returning:
- Center `(x, y)`
- Gauge radius

This center becomes the reference point for all subsequent needle and scale detection.

### 2. Scale / Tick Detection
Tick marks are detected from the white markings around the gauge:
1. Convert the image to grayscale and HSV color space.
2. Create a mask for bright, low-saturation pixels.
3. Analyze the gauge region at different angles.
4. Detect peaks corresponding to visible tick marks.
5. Store the detected tick angles.

### 3. Needle Detection
The needle is identified using color-based segmentation combined with Hough Line detection. The line closest to the gauge center and aligned with the radial direction is selected as the needle, and its angle is calculated relative to the center.

### 4. RPM Calculation
Once the needle angle and tick angles are known, the system determines where the needle falls between two consecutive ticks using interpolation rather than simply snapping to the nearest tick:

```
position = lower_tick + fraction_between_ticks
```

For this gauge, the visible scale represents **0–30 × 100 RPM**, so each scale interval equals **100 RPM**.

### 5. Output Precision
The final RPM value is rounded to the nearest 10:

| Raw Value | Rounded RPM |
|-----------|-------------|
| 2468.68   | 2470        |
| 2376.40   | 2380        |
| 1981.23   | 1980        |
| 902.66    | 900         |

---

## Project Structure

```text
HIL-Test-Automation/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── needle_gauge/
│       └── input images
│
├── output/
│   └── generated result images
│
└── src/
    ├── gauge_detector.py
    ├── needle_detector.py
    ├── scale_detector.py
    └── rpm_calculator.py
```

---

## Requirements

- Python 3.10+
- OpenCV (`opencv-python`)
- NumPy
- SciPy

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/IshankSingla/HIL-Test-Automation.git
```

**2. Navigate to the project**
```bash
cd HIL-Test-Automation
```

**3. Create a virtual environment**
```bash
python -m venv venv
```

**4. Activate the virtual environment**

PowerShell:
```bash
venv\Scripts\Activate.ps1
```

Command Prompt:
```bash
venv\Scripts\activate
```

**5. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## Usage

Place your input images inside:

```text
data/needle_gauge/
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`

Example:
```text
data/
└── needle_gauge/
    ├── image1.jpg
    ├── image2.jpg
    ├── image3.jpg
    └── image4.jpg
```

Then run:

```bash
python main.py
```

The program automatically detects and processes **all** supported images in the input directory — no need to specify filenames manually.

### Example Terminal Output

```text
==================================================
Processing: data/needle_gauge/image1.jpg
==================================================

Gauge center: (500, 434)
Gauge radius: 360
Needle angle: 345.03 degrees
Detected ticks: 31
Scale start angle: 154.90
Scale end angle: 385.20
Gauge reading: 2470.00 RPM

Result saved to: output/image1_result.jpg
```

---

## Output

Processed images are saved automatically inside:

```text
output/
├── image1_result.jpg
├── image2_result.jpg
└── image3_result.jpg
```

Each output image includes:
- Detected gauge boundary
- Detected gauge center
- Detected needle line
- Detected needle angle
- Calculated RPM value

---

## Error Handling

If an image cannot be processed, the program logs the error and continues with the remaining images:

```text
ERROR processing image3.jpg: Could not detect gauge tick marks.
```

---

## Design Principle

> The system never uses image filenames to determine RPM values.

There is **no** logic such as:

```python
if filename == "image1.jpg":
    rpm = 900
```

Instead, every reading is computed as:

```
Gauge Geometry + Detected Tick Positions + Detected Needle Position = RPM
```

This means input images can be swapped out freely without ever touching the code.

---

## Limitations

- Very low-quality images may reduce detection accuracy.
- Heavy reflections or shadows may interfere with tick detection.
- A needle of a significantly different color may require recalibrating color segmentation.
- A significantly different gauge design may require recalibration.
- Gauges with a different numerical scale need separate calibration.
- Current implementation assumes a scale of **0–30 × 100 RPM**.

---

## Future Improvements

- More robust needle centerline detection
- Automatic gauge label detection via OCR
- Support for additional gauge designs
- Automatic scale calibration
- Perspective correction for tilted images
- Improved handling of reflections and shadows
- Confidence score per RPM prediction
- Real-time video/stream processing
- Unit tests for individual detection modules

---

## Modules

| Module | Description |
|---|---|
| `main.py` | Controls the complete pipeline and performs batch image processing |
| `src/gauge_detector.py` | Detects the gauge circle, center, and radius |
| `src/needle_detector.py` | Detects the colored needle and calculates its angle |
| `src/scale_detector.py` | Detects the gauge scale and tick positions |
| `src/rpm_calculator.py` | Converts detected needle position into an RPM value |

---

## Repository

[https://github.com/IshankSingla/HIL-Test-Automation](https://github.com/IshankSingla/HIL-Test-Automation)