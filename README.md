# Liquid Level Detection in Bottles 🧪📏

This project presents **five different approaches** to detect **liquid levels in bottles** using image processing techniques with Python and OpenCV.

## 🔍 Methods Overview

The repository includes five separate scripts, each employing a distinct strategy to locate the horizontal liquid surface inside transparent or semi-transparent bottles.

### 1. `color_diff.py` – Pixel Intensity Analysis
- Converts the image to grayscale.
- Binarizes and normalizes the image.
- Calculates **row-wise pixel intensity** (Y-axis).
- Visualizes the intensity distribution as a bar plot.

📊 Good for: Even lighting and clear air-liquid contrast.

---

### 2. `line_overlay.py` – Sobel + Morphological Filters
- Applies **Sobel Y edge detection** to enhance horizontal edges.
- Uses **morphological closing and opening** to refine those edges.
- Applies **Hough Line Transform** to detect straight horizontal lines.

🧼 Good for: Sharp edges and images with minimal noise.

---

### 3. `line_detection_draw.py` – Hybrid Analysis
- Combines Sobel, morphology, and **row-wise intensity summation**.
- Outputs both processed images and **bar plots** of pixel intensity.
- Includes intermediate visualizations for debugging and evaluation.

🧪 Good for: Complex scenes or noisy backgrounds.

---

### 4. `y_coordinate.py` – Horizontal Line Clustering
- Extracts horizontal lines using Sobel + Canny + Hough.
- Gathers the **Y-coordinates** of these lines.
- Applies **KMeans clustering** to group and highlight representative liquid levels.

📍 Good for: Precise line-level Y-coordinate localization.

---

### 5. `a_b_params.py` – Line Equation Clustering
- Detects horizontal-ish lines and converts them to **slope-intercept (a, b)** form.
- Clusters these lines using **KMeans** to group dominant line patterns.

📐 Good for: Mathematical analysis and grouping of similar line orientations.

---

## 📁 Folder Structure

```
liquid-level-detection/
│
├── a_b_params.py # Slope-intercept (a, b) clustering of horizontal lines
├── y_coordinate.py # KMeans clustering of horizontal Y-coordinates
├── line_overlay.py # Sobel + Morphology + Hough Line method
├── color_diff.py # Grayscale pixel intensity difference method
├── line_detection_draw.py # Combined hybrid method
│
├── requirements.txt # Python dependencies
├── README.md # Project documentation
│
├── img/ # Input images folder
│ └── (your .jpg/.png files here)
│
├── output_folder/ # Output from line_overlay, y_coordinate, a_b_params
│ └── (processed images/results)
│
└── color_diff/ # Output from color_diff.py
├── row_sums_X.jpeg
└── row_sums_X.txt
```