# Liquid Level Detection in Bottles 🧪📏

This project presents three different approaches to detect **liquid levels in bottles** using image processing techniques with Python and OpenCV.

## 🔍 Methods Overview

The repository includes three separate scripts, each employing a distinct approach to detect the horizontal liquid line (surface) inside transparent or semi-transparent bottles.

### 1. `color_diff.py` – Pixel Intensity Analysis
- Converts the image to grayscale.
- Binarizes and normalizes the image.
- Calculates **row-wise pixel intensity** (Y-axis).
- Visualizes the intensity distribution to locate the liquid level.

📊 Good for: Consistent lighting, clear separation between liquid and air.

---

### 2. `line_overlay.py` – Sobel + Morphological Filters
- Applies vertical **Sobel edge detection** to highlight horizontal transitions.
- Uses **morphological closing and opening** to enhance continuous edges.
- Employs **Hough Line Transform** to detect horizontal lines.

🧼 Good for: Bottles with sharp liquid-air edges and minor noise.

---

### 3. `line_detection_draw.py` – Hybrid Approach
- Combines Sobel + morphology with **row-wise pixel analysis**.
- Normalizes the output and creates a **density bar plot**.
- Visual inspection via intermediate steps using `cv.imshow()`.

🧪 Good for: Complex backgrounds or cases where multiple techniques are needed.

---

## 📁 Folder Structure

```
liquid-level-detection/
│
├── line_overlay.py          # Sobel + Morphology + Hough Line method
├── color_diff.py            # Grayscale pixel intensity difference method
├── line_detection_draw.py   # Combined hybrid method
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
│
├── img/                     # Input images folder
│   └── (your .jpg/.png files here)
│
├── output_folder/            # Output from line_detection_draw and line_overlay
│   └── (processed images/results)
│
└── color_diff/              # Output from color_diff.py
    ├── satir_toplamlari_X.jpeg
    └── satir_toplamlari_X.txt
```
