import cv2
import numpy as np
from skimage.filters import threshold_sauvola

img_path = r"data/binarised_representative_samples/tamil_stone/tamil_026_original.jpg"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
H, W = gray.shape

configs = [
    (15, 0.05),
    (15, 0.15),
    (15, 0.25),
    (25, 0.15),
    (25, 0.25),
    (35, 0.20),
    (35, 0.30),
    (51, 0.20),
    (51, 0.30),
    (51, 0.40),
    (61, 0.30),
    (61, 0.40)
]

for ws, k in configs:
    thresh = threshold_sauvola(gray, window_size=ws, k=k)
    binary = (gray < thresh).astype(np.uint8) * 255
    # Morph close
    binary_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_close, connectivity=8)
    areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
    areas.sort(reverse=True)
    
    white_pct = (binary == 255).mean() * 100
    top_3_areas = areas[:3]
    print(f"ws={ws}, k={k:.2f} | white={white_pct:.2f}% | components={num_labels} | top areas={top_3_areas}")
