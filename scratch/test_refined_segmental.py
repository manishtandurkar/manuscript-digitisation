import cv2
import numpy as np
from skimage.filters import threshold_sauvola

img_path = "data/binarised_representative_samples/tamil_stone/tamil_026_original.jpg"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
H, W = gray.shape

# 1. Sauvola thresholding with ws=15, k=0.25 (keeps components small & isolated)
thresh = threshold_sauvola(gray, window_size=15, k=0.25)
binary = (gray < thresh).astype(np.uint8) * 255

# 2. Morphology close (2x2) to connect characters
binary_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

# 3. Connected Components Analysis segment-by-segment filtering
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_close, connectivity=8)

canvas = np.zeros_like(binary_close)
min_area = 12
max_area = 2000

for label in range(1, num_labels):
    area = int(stats[label, cv2.CC_STAT_AREA])
    if min_area <= area <= max_area:
        canvas[labels == label] = 255

# 4. Flood fill from borders to wipe out scanner noise and slanted stone boundaries touching edges
flood_mask = np.zeros((H + 2, W + 2), np.uint8)
for x in range(W):
    for y in [0, H - 1]:
        if canvas[y, x] == 255:
            cv2.floodFill(canvas, flood_mask, (x, y), 0)
for y in range(H):
    for x in [0, W - 1]:
        if canvas[y, x] == 255:
            cv2.floodFill(canvas, flood_mask, (x, y), 0)

# 5. Invert to produce black characters on white background
canvas_inv = cv2.bitwise_not(canvas)

# Save
cv2.imwrite("scratch/segmental_clean_output.png", canvas_inv)

# Let's count white and black pixels
total = canvas_inv.size
black = np.sum(canvas_inv == 0)
white = np.sum(canvas_inv == 255)
print(f"Segmental Clean Output: black={black} ({black/total*100:.2f}%), white={white} ({white/total*100:.2f}%)")
