import os
from PIL import Image
import numpy as np
import glob

match_case = "*_BASNet-PVT-ssp-3.png"

def check_black_images(image_dir):
    """
    Checks if images in a directory are all black.

    Args:
        image_dir: The directory containing the images.

    Returns:
        A list of filenames (including path) of non-zero images.  Returns an empty list if all are zero.
    """

    image_files = glob.glob(os.path.join(image_dir, match_case))
    non_zero_images = []

    if not image_files:
      print(f"No files found matching the pattern in {image_dir}")
      return non_zero_images #Return empty list if no files found

    for image_file in image_files:
        try:
            img = Image.open(image_file).convert("L") # Convert to grayscale for faster processing and to handle potential different color channels
            img_array = np.array(img)

            if np.any(img_array != 0):  # Check if any pixel is not zero
                non_zero_images.append(image_file)
        except Exception as e:
            print(f"Error processing image {image_file}: {e}")

    return non_zero_images


if __name__ == "__main__":
    image_directory = "/home/shenkunye/GangZheng/dataset/prediction/BASNet-PVT/SSP2000/test"
    non_zero_images = check_black_images(image_directory)

    if non_zero_images:
        print("The following images are NOT all black:")
        for image_file in non_zero_images:
            print(image_file)
    else:
        print("All images in the directory are black (all zeros).")