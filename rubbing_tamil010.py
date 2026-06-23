import cv2
from pathlib import Path

img_path = r"C:\6th semester EL's\Interdisciplinary project\Implementation\manuscript-digitisation\data\binarised_representative_samples\tamil_stone\tamil_010_original.jpg"
img = cv2.imread(img_path)

if img is None:
    print(f"Error: could not load image at {img_path}")
else:
    print("Shape:", img.shape)
