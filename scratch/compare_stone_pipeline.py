import cv2
import numpy as np
from src.binarise import binarise_stone

img_path = "data/binarised_representative_samples/tamil_stone/tamil_026_original.jpg"
img = cv2.imread(img_path)

if img is None:
    print("Could not load input image")
    exit(1)

# Run binarise_stone
bin_stone = binarise_stone(img)

# Load pipeline image
pipeline_path = "data/binarised/tamil_stone__tamil_026_jpg_binarised.png"
pipeline_img = cv2.imread(pipeline_path, cv2.IMREAD_GRAYSCALE)

if pipeline_img is not None:
    # Check if they are identical
    diff = cv2.absdiff(bin_stone, pipeline_img)
    non_zero = np.count_nonzero(diff)
    print(f"Difference pixels: {non_zero} (out of {diff.size})")
    if non_zero == 0:
        print("They are identical!")
    else:
        # Check if they are inverted
        diff_inv = cv2.absdiff(cv2.bitwise_not(bin_stone), pipeline_img)
        non_zero_inv = np.count_nonzero(diff_inv)
        print(f"Difference when inverted: {non_zero_inv}")
else:
    print("Pipeline image not found")
