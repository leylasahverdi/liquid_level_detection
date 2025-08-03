import cv2 as cv
import numpy as np
import os
from sklearn.cluster import KMeans


# Function to display image in a resizable window
def show_img(image):
    cv.namedWindow('image', cv.WINDOW_NORMAL)
    cv.imshow('image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()


# Define input and output folder paths
input_folder = r"input_folder_path"
output_folder = r"output_folder_path"
os.makedirs(output_folder, exist_ok=True)

# Process each image
for i, file_name in enumerate(os.listdir(input_folder), start=1):
    file_path = os.path.join(input_folder, file_name)

    if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        full_path = os.path.join(input_folder, file_name)
        img = cv.imread(full_path)

        if img is not None:
            # Convert to grayscale and blur
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            blurred = cv.GaussianBlur(gray, (3, 3), 0)
            show_img(blurred)

            # Edge detection
            edges = cv.Canny(blurred, threshold1=50, threshold2=150)
            _, binary = cv.threshold(edges, 50, 255, cv.THRESH_BINARY)
            show_img(binary)

            # Find and draw contours
            contours, _ = cv.findContours(binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            contour_img = np.zeros_like(img)
            cv.drawContours(contour_img, contours, contourIdx=-1, thickness=2, color=(255, 255, 255))
            show_img(contour_img)

            # Convert to grayscale and apply morphological ops
            contour_gray = cv.cvtColor(contour_img.astype(np.uint8), cv.COLOR_BGR2GRAY)
            dilated = cv.dilate(contour_gray, None, iterations=10)
            eroded = cv.erode(dilated, None, iterations=8)
            show_img(eroded)

            # Save intermediate result
            save_path = os.path.join(output_folder, f"eroded_{i}.jpeg")
            cv.imwrite(save_path, eroded)

            # Reload and apply Sobel Y filter for horizontal edges
            gray = cv.imread(save_path, cv.IMREAD_GRAYSCALE)
            sobel_y = cv.Sobel(gray, cv.CV_64F, dx=0, dy=1, ksize=3)
            sobel_y = cv.convertScaleAbs(sobel_y)
            show_img(sobel_y)

            # Threshold to create horizontal mask
            _, horizontal_mask = cv.threshold(sobel_y, 50, 255, cv.THRESH_BINARY)
            show_img(horizontal_mask)

            # Apply mask to Canny edges
            edges = cv.Canny(gray, 50, 150)
            horizontal_edges = cv.bitwise_and(edges, horizontal_mask)

            # Use Hough transform to detect horizontal lines
            lines = cv.HoughLinesP(
                horizontal_edges,
                rho=1,
                theta=np.pi / 180,
                threshold=50,
                minLineLength=50,
                maxLineGap=20
            )
            print(lines)

            # Extract Y coordinates of nearly horizontal lines
            y_coords = []
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if abs(y1 - y2) < 5:  # Allow small slope
                        y_coords.append((y1 + y2) // 2)

            # Cluster the Y coordinates using KMeans
            if y_coords:
                y_coords_np = np.array(y_coords).reshape(-1, 1)
                kmeans = KMeans(n_clusters=3, random_state=0).fit(y_coords_np)
                centers = kmeans.cluster_centers_.astype(int).flatten()

                # Draw clustered lines on the original image
                output_img = img.copy()
                for y in centers:
                    cv.line(output_img, (0, y), (output_img.shape[1], y), 255, 2)

                save_path = os.path.join(output_folder, f"eroded_{i}.jpeg")
                cv.imwrite(save_path, output_img)
                show_img(output_img)
            else:
                print("Not enough lines found.")
        else:
            print(f"Failed to load image: {file_path}")
