import cv2
import numpy as np
from skimage.filters import threshold_sauvola

img_path = r"data/binarised_representative_samples/tamil_stone/tamil_026_original.jpg"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ws = 15
k = 0.05
thresh = threshold_sauvola(gray, window_size=ws, k=k)
binary = (gray < thresh).astype(np.uint8) * 255
binary_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_close, connectivity=8)
print(f"Total labels: {num_labels}")
areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
areas.sort(reverse=True)
print(f"Top 15 component areas: {areas[:15]}")
