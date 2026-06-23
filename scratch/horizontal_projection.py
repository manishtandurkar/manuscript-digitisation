import cv2
import numpy as np

img = cv2.imread(r"C:\Users\nanda_4h6zihz\.gemini\antigravity-ide\brain\9bb206b2-f39c-4a1f-befa-83279631baab\media__1782219053166.png", cv2.IMREAD_GRAYSCALE)
H, W = img.shape

# Threshold to binary (0 for black/characters, 255 for white/bg)
_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Sum of black pixels per column (255 - thresh) / 255
col_black_sums = np.sum((255 - thresh) // 255, axis=0)

# Print a simple horizontal projection map
print("Horizontal projection of black pixels (per column):")
for group in range(0, W, 10):
    group_end = min(W, group + 10)
    avg_black = np.mean(col_black_sums[group:group_end])
    print(f"Cols {group:03d}-{group_end-1:03d}: {'#' * int(avg_black/2)}")
