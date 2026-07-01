import cv2
import numpy as np

target_path = r"C:\Users\nanda_4h6zihz\.gemini\antigravity-ide\brain\9bb206b2-f39c-4a1f-befa-83279631baab\media__1782219053166.png"
img = cv2.imread(target_path)
if img is None:
    print("Failed to load target image.")
    exit(1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect horizontal and vertical lines, or find bounding boxes of contours
_, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

print(f"Target shape: {img.shape}")
print(f"Number of components in target thresh (>240 inv): {num_labels}")

# Sort stats by area
sorted_idx = np.argsort(-stats[:, cv2.CC_STAT_AREA])
for i in range(1, min(10, num_labels)):
    idx = sorted_idx[i]
    x = stats[idx, cv2.CC_STAT_LEFT]
    y = stats[idx, cv2.CC_STAT_TOP]
    w = stats[idx, cv2.CC_STAT_WIDTH]
    h = stats[idx, cv2.CC_STAT_HEIGHT]
    area = stats[idx, cv2.CC_STAT_AREA]
    print(f"Component {i}: x={x}, y={y}, w={w}, h={h}, area={area}")
