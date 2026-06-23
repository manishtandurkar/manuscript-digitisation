import cv2
import numpy as np

# Load target
target = cv2.imread(r"C:\Users\nanda_4h6zihz\.gemini\antigravity-ide\brain\9bb206b2-f39c-4a1f-befa-83279631baab\media__1782219053166.png", cv2.IMREAD_GRAYSCALE)
H_t, W_t = target.shape

# Load grid best
img = cv2.imread("scratch/grid_best_inv.png", cv2.IMREAD_GRAYSCALE)

# Resize to target size using different interpolation methods
for interp, name in [
    (cv2.INTER_LINEAR, "linear"),
    (cv2.INTER_CUBIC, "cubic"),
    (cv2.INTER_LANCZOS4, "lanczos"),
    (cv2.INTER_NEAREST, "nearest")
]:
    img_r = cv2.resize(img, (W_t, H_t), interpolation=interp)
    
    # Compute similarity at various thresholds
    for th in [100, 127, 150, 180, 200, 220, 240]:
        _, img_bin = cv2.threshold(img_r, th, 255, cv2.THRESH_BINARY)
        _, target_bin = cv2.threshold(target, th, 255, cv2.THRESH_BINARY)
        match = np.sum(img_bin == target_bin)
        sim = match / target_bin.size
        if sim > 0.70:
            print(f"Interp: {name} | Threshold: {th} | Similarity: {sim:.4f}")
