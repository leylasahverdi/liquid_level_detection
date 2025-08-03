import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# Utility function to display image
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
        full_path = os.path.join(input_folder, file_name)
        img = cv.imread(full_path, cv.IMREAD_GRAYSCALE)

        if img is not None:
            # Apply binary threshold to enhance contrast
            _, binary_img = cv.threshold(img, 150, 255, cv.THRESH_BINARY)

            # Normalize image to range [0, 1]
            normalized = binary_img / 255

            # Calculate row-wise pixel intensity (summed over columns)
            row_sums = np.sum(normalized, axis=1)

            # Optionally scale values for consistency
            scaler = StandardScaler()
            scaled = scaler.fit_transform(row_sums.reshape(-1, 1)).flatten()

            # Plot and visualize pixel intensity by row
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(row_sums)), row_sums)
            plt.xlabel("Row (Y-axis)")
            plt.ylabel("Total Pixel Value")
            plt.title("Row-wise Pixel Intensity")
            plt.tight_layout()
            plt.show()

            # Display the normalized binary image
            show_img(normalized)

            # Save the plot to output folder
            plot_path = os.path.join(output_folder, f"row_sums_{i}.jpeg")
            plt.savefig(plot_path)
            plt.close()

            # Save raw row sums as text data
            data_path = os.path.join(output_folder, f"row_sums_{i}.txt")
            np.savetxt(data_path, row_sums)

        else:
            print("Could not process the image.")
    else:
        print(f"Unsupported file format: {file_path}")
