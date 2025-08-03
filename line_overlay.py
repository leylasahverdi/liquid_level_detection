import cv2 as cv
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


# Utility function to display images
def show_img(image, winname='image'):
    cv.namedWindow(winname, cv.WINDOW_NORMAL)
    cv.imshow(winname, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


# Define input and output folders
input_folder = r"input_folder_path"
output_folder = r"output_folder_path"

# Loop through each image in the input folder
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

            # Apply vertical Sobel filter (to detect horizontal edges)
            sobel_y = cv.Sobel(blurred, cv.CV_64F, dx=0, dy=1, ksize=3)
            sobel_y = cv.convertScaleAbs(sobel_y)

            # Apply binary thresholding
            _, binary = cv.threshold(sobel_y, 150, 255, cv.THRESH_BINARY)

            # Define a horizontal kernel and apply morphological closing
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (30, 1))
            morphed = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
            show_img(morphed)

            # Detect horizontal lines using Hough Transform
            lines = cv.HoughLinesP(morphed, 1, np.pi / 180, threshold=30, minLineLength=40, maxLineGap=10)

            # Convert grayscale image back to color to draw lines
            color_img = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv.line(color_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw green lines
                show_img(color_img)
            else:
                print("No lines were found.")

            # Apply morphological opening to remove small noise
            opened = cv.morphologyEx(morphed, cv.MORPH_OPEN, kernel)

            # Apply closing again to reconnect broken lines
            final_result = cv.morphologyEx(opened, cv.MORPH_CLOSE, kernel)

            # Display the final result
            show_img(final_result)
