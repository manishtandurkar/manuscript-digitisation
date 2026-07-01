import cv2
import numpy as np

# Load target
target = cv2.imread(r"C:\Users\nanda_4h6zihz\.gemini\antigravity-ide\brain\9bb206b2-f39c-4a1f-befa-83279631baab\media__1782218721649.png", cv2.IMREAD_GRAYSCALE)
target_crop = target[6:360, 4:506]
_, target_bin = cv2.threshold(target_crop, 127, 255, cv2.THRESH_BINARY)
H_tc, W_tc = target_crop.shape

# Load grid best
img = cv2.imread("scratch/grid_best_inv.png", cv2.IMREAD_GRAYSCALE)

# Resize to target crop
img_r = cv2.resize(img, (W_tc, H_tc), interpolation=cv2.INTER_NEAREST)
_, img_bin = cv2.threshold(img_r, 127, 255, cv2.THRESH_BINARY)

match = np.sum(img_bin == target_bin)
sim = match / target_bin.size
print(f"Similarity against media__1782218721649.png crop: {sim:.4f}")
