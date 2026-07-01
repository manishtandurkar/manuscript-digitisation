import cv2
import numpy as np
from pathlib import Path
from skimage.filters import threshold_sauvola

img_path = r"data/binarised_representative_samples/tamil_stone/tamil_026_original.jpg"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
H, W = gray.shape[:2]

# Step 2: Sauvola
ws = 15
k = 0.05
thresh = threshold_sauvola(gray, window_size=ws, k=k)
binary = (gray < thresh).astype(np.uint8) * 255
print(f"After Sauvola: white pixels = {np.sum(binary == 255)}")

# Step 3: Morph close
binary_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
print(f"After Morph Close: white pixels = {np.sum(binary_close == 255)}")

# Step 4: CCA
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_close, connectivity=8)
canvas = np.zeros((H, W), dtype=np.uint8)
for label in range(1, num_labels):
    area = int(stats[label, cv2.CC_STAT_AREA])
    if 1 <= area <= 50000:
        canvas[labels == label] = 255
print(f"After CCA reconstruction: white pixels = {np.sum(canvas == 255)}")

# Step 5: Corner floodfill
flood_mask = np.zeros((H + 2, W + 2), np.uint8)
canvas_flood = canvas.copy()
for corner in [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]:
    if canvas_flood[corner[0], corner[1]] == 255:
        print(f"Flood filling from corner {corner}")
        cv2.floodFill(canvas_flood, flood_mask, (corner[1], corner[0]), 0)
print(f"After Flood Fill: white pixels = {np.sum(canvas_flood == 255)}")

canvas_inverted = cv2.bitwise_not(canvas_flood)
print(f"Final Inverted: black pixels = {np.sum(canvas_inverted == 0)}")
