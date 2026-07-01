import cv2
import numpy as np
from skimage.filters import threshold_sauvola

img_path = "data/raw/tamil_stone/IMG_3924.jpg"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
H, W = gray.shape

# Crop central region
cy, cx = H // 2, W // 2
crop = gray[cy-400:cy+400, cx-400:cx+400]

# Step 1: Bilateral Filter
denoised = cv2.bilateralFilter(crop, d=9, sigmaColor=50, sigmaSpace=50)
cv2.imwrite("scratch/img3924_crop_denoised.jpg", denoised)

# Step 2: Sauvola ws=151, k=0.15
thresh = threshold_sauvola(denoised, window_size=151, k=0.15)
binary = (denoised < thresh).astype(np.uint8) * 255
cv2.imwrite("scratch/img3924_crop_sauvola_default.png", binary)
print(f"Sauvola default: white pixels = {np.sum(binary == 255)}")

# Step 3: Morph close 3x3
binary_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
cv2.imwrite("scratch/img3924_crop_close_default.png", binary_close)
print(f"Morph close: white pixels = {np.sum(binary_close == 255)}")

# Step 4: Remove noise blobs with min_size=900, min_length=37
def remove_noise(binary, min_size, min_length):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    kept_count = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area >= min_size or max(cw, ch) >= min_length:
            cleaned[labels == label] = 255
            kept_count += 1
    print(f"Components kept: {kept_count} out of {num_labels}")
    return cleaned

cleaned = remove_noise(binary_close, 900, 37)
cv2.imwrite("scratch/img3924_crop_cleaned_default.png", cleaned)
print(f"Cleaned output: white pixels = {np.sum(cleaned == 255)}")
