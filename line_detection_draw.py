import cv2 as cv
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from scipy.stats import zscore
from itertools import groupby
from operator import itemgetter
from sklearn.preprocessing import StandardScaler


# Utility function to display an image
def show_img(image, winname='image'):
    cv.namedWindow(winname, cv.WINDOW_NORMAL)
    cv.imshow(winname, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


# Define input and output folder paths
input_folder = r"input_folder_path"
output_folder = r"output_folder_path"

# Process each image in the input folder
for i, file_name in enumerate(os.listdir(input_folder), start=1):
    file_path = os.path.join(input_folder, file_name)

    if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        full_path = os.path.join(input_folder, file_name)
        img = cv.imread(full_path)

        if img is not None:
            # Convert to grayscale
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            # Apply Gaussian blur to reduce noise
            blurred = cv.GaussianBlur(gray, (7, 7), 0)
            show_img(blurred)

            # Apply vertical Sobel filter to highlight horizontal edges
            sobel_y = cv.Sobel(blurred, cv.CV_64F, dx=0, dy=1, ksize=3)
            sobel_y = cv.convertScaleAbs(sobel_y)

            # Threshold to create binary image
            _, binary = cv.threshold(sobel_y, 150, 255, cv.THRESH_BINARY)

            # Apply morphological closing with a wide horizontal kernel
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (30, 1))
            morphed = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
            show_img(morphed)

            # Re-threshold after morphology and normalize
            _, rethresholded = cv.threshold(morphed, 150, 255, cv.THRESH_BINARY)
            normalized = rethresholded / 255

            # Calculate sum of white pixels per row (intensity by Y-axis)
            row_sums = np.sum(normalized, axis=1)

            # Plot row-wise intensity
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(row_sums)), row_sums)
            plt.xlabel("Row (Y Axis)")
            plt.ylabel("Total Pixel Value")
            plt.title("Row-wise Pixel Intensity")
            plt.tight_layout()
            plt.show()

            # Show the normalized binary image
            show_img(normalized)
