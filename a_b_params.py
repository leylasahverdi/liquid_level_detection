import cv2 as cv
import numpy as np
import os
from sklearn.cluster import KMeans


# Function to display images
def show_img(image):
    cv.namedWindow('image', cv.WINDOW_NORMAL)
    cv.imshow('image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()


# Input and output folder paths
input_folder = r"input_folder_path"
output_folder = r"output_folder_path"
os.makedirs(output_folder, exist_ok=True)

# Process each image in the input folder
for i, file_name in enumerate(os.listdir(input_folder), start=1):
    file_path = os.path.join(input_folder, file_name)

    if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(input_folder, file_name)
        img = cv.imread(img_path)

        if img is not None:
            # Convert to grayscale and apply Gaussian blur
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            blurred = cv.GaussianBlur(gray, (3, 3), 0)

            # Detect edges using Canny
            edges = cv.Canny(blurred, threshold1=50, threshold2=150)

            # Apply binary threshold
            _, binary = cv.threshold(edges, 50, 255, cv.THRESH_BINARY)
            show_img(binary)

            # Find contours
            contours, _ = cv.findContours(binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

            # Draw contours on a blank image
            contour_img = np.zeros_like(img)
            cv.drawContours(contour_img, contours, contourIdx=-1, thickness=2, color=(255, 255, 255))
            show_img(contour_img)

            # Convert to grayscale and apply morphological operations
            contour_gray = cv.cvtColor(contour_img.astype(np.uint8), cv.COLOR_BGR2GRAY)
            dilated = cv.dilate(contour_gray, None, iterations=10)
            eroded = cv.erode(dilated, None, iterations=8)
            show_img(eroded)

            # Save preprocessed image
            save_path = os.path.join(output_folder, f"eroded_{i}.jpeg")
            cv.imwrite(save_path, eroded)

            # Reload image in grayscale for further processing
            gray = cv.imread(save_path, cv.IMREAD_GRAYSCALE)

            # Detect horizontal gradients using Sobel Y
            sobel_y = cv.Sobel(gray, cv.CV_64F, dx=0, dy=1, ksize=3)
            sobel_y = cv.convertScaleAbs(sobel_y)
            show_img(sobel_y)

            # Threshold to create horizontal mask
            _, horizontal_mask = cv.threshold(sobel_y, 50, 255, cv.THRESH_BINARY)
            show_img(horizontal_mask)

            # Edge detection again for combination
            edges = cv.Canny(gray, 50, 150)

            # Mask to get only horizontal lines
            horizontal_edges = cv.bitwise_and(edges, horizontal_mask)

            # Use Probabilistic Hough Transform to find lines
            lines = cv.HoughLinesP(
                horizontal_edges,
                rho=1,
                theta=np.pi / 180,
                threshold=50,
                minLineLength=50,
                maxLineGap=20
            )

            print(lines)

            output_image = np.zeros_like(gray)
            line_params = []

            # Filter and store slope-intercept form of horizontal-ish lines
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if abs(y1 - y2) < 10:  # allow slight slope
                        a = (y2 - y1) / (x2 - x1 + 1e-6)
                        b = y1 - a * x1
                        line_params.append([a, b])

            # Cluster the lines using KMeans
            if line_params:
                param_np = np.array(line_params)
                kmeans = KMeans(n_clusters=3, random_state=0).fit(param_np)
                centers = kmeans.cluster_centers_

                output_image = img.copy()
