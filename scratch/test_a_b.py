import cv2
import numpy as np
from skimage.filters import threshold_sauvola

img_path = "data/binarised_representative_samples/tamil_stone/tamil_026_original.jpg"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Method A: default binarise_stone parameters
# 1. Bilateral
denoised_a = cv2.bilateralFilter(gray, d=5, sigmaColor=30, sigmaSpace=30)
# 2. Sauvola ws=25, k=0.12
thresh_a = threshold_sauvola(denoised_a, window_size=25, k=0.12)
bin_a = (denoised_a < thresh_a).astype(np.uint8) * 255
# 3. Morph close
bin_a_close = cv2.morphologyEx(bin_a, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
# 4. Remove noise blobs
def remove_noise(binary, min_size, min_length):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area >= min_size or max(cw, ch) >= min_length:
            cleaned[labels == label] = 255
    return cleaned

bin_a_clean = remove_noise(bin_a_close, 12, 6)
# 5. Flood fill borders
h_b, w_b = bin_a_clean.shape
flood_mask = np.zeros((h_b + 2, w_b + 2), np.uint8)
for x in range(w_b):
    for y in [0, h_b - 1]:
        if bin_a_clean[y, x] == 255:
            cv2.floodFill(bin_a_clean, flood_mask, (x, y), 0)
for y in range(h_b):
    for x in [0, w_b - 1]:
        if bin_a_clean[y, x] == 255:
            cv2.floodFill(bin_a_clean, flood_mask, (x, y), 0)

# Save inverted
cv2.imwrite("scratch/test_method_a_inv.png", cv2.bitwise_not(bin_a_clean))


# Method B: Sauvola ws=15, k=0.25 (No bilateral filter, keeping fine details)
thresh_b = threshold_sauvola(gray, window_size=15, k=0.25)
bin_b = (gray < thresh_b).astype(np.uint8) * 255
bin_b_close = cv2.morphologyEx(bin_b, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
# Apply same cleanup
bin_b_clean = remove_noise(bin_b_close, 12, 6)
# Flood fill borders
flood_mask_b = np.zeros((h_b + 2, w_b + 2), np.uint8)
for x in range(w_b):
    for y in [0, h_b - 1]:
        if bin_b_clean[y, x] == 255:
            cv2.floodFill(bin_b_clean, flood_mask_b, (x, y), 0)
for y in range(h_b):
    for x in [0, w_b - 1]:
        if bin_b_clean[y, x] == 255:
            cv2.floodFill(bin_b_clean, flood_mask_b, (x, y), 0)

# Save inverted
cv2.imwrite("scratch/test_method_b_inv.png", cv2.bitwise_not(bin_b_clean))

print("Saved test outputs for Method A and Method B.")
