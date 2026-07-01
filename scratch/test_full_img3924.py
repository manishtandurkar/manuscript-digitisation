import cv2
import numpy as np
from skimage.filters import threshold_sauvola

img_path = "data/raw/tamil_stone/IMG_3924.jpg"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
H, W = gray.shape
shorter = min(H, W)

# 1. Median filter (ksize = (shorter // 100) | 1 = 31)
ksize = (shorter // 100) | 1
blurred = cv2.medianBlur(gray, ksize)

# 2. Sauvola (ws = (shorter // 30) | 1 = 101, k = 0.25)
ws = (shorter // 30) | 1
k = 0.25
thresh = threshold_sauvola(blurred, window_size=ws, k=k)
binary = (blurred < thresh).astype(np.uint8) * 255

# 3. Morph close (3x3)
binary_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

# 4. Remove noise blobs
# min_size = (shorter // 150) ** 2 = 400
# min_length = shorter // 120 = 25
min_size = (shorter // 150) ** 2
min_length = shorter // 120

num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_close, connectivity=8)
cleaned = np.zeros_like(binary_close)
kept_count = 0
for label in range(1, num_labels):
    area = int(stats[label, cv2.CC_STAT_AREA])
    cw = int(stats[label, cv2.CC_STAT_WIDTH])
    ch = int(stats[label, cv2.CC_STAT_HEIGHT])
    if area >= min_size or max(cw, ch) >= min_length:
        cleaned[labels == label] = 255
        kept_count += 1

print(f"Whole image: kept {kept_count} out of {num_labels} components.")

# 5. Flood fill borders
flood_mask = np.zeros((H + 2, W + 2), np.uint8)
for x in range(W):
    for y in [0, H - 1]:
        if cleaned[y, x] == 255:
            cv2.floodFill(cleaned, flood_mask, (x, y), 0)
for y in range(H):
    for x in [0, W - 1]:
        if cleaned[y, x] == 255:
            cv2.floodFill(cleaned, flood_mask, (x, y), 0)

# Save the final binarised output
output_path = "scratch/IMG_3924_binarised_segmental.png"
cv2.imwrite(output_path, cleaned)
white_pct = (cleaned == 255).mean() * 100
print(f"Binarised output white %: {white_pct:.2f}%")
