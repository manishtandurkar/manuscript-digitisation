import cv2
import numpy as np
from pathlib import Path

target = cv2.imread(r"C:\Users\nanda_4h6zihz\.gemini\antigravity-ide\brain\9bb206b2-f39c-4a1f-befa-83279631baab\media__1782219053166.png", cv2.IMREAD_GRAYSCALE)
_, target_bin = cv2.threshold(target, 127, 255, cv2.THRESH_BINARY)

files = [
    "scratch/sauvola_ws15_k25_inv.png",
    "scratch/sauvola_ws25_k25_inv.png",
    "scratch/sauvola_ws35_k30_inv.png",
    "scratch/sauvola_ws51_k40_inv.png",
]

for f in files:
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    max_sim = 0
    best_w = 0
    best_x = 0
    # Test different resize widths to account for scaling differences
    for w in [509, 531, 581, 632]:
        h = 436
        img_r = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        _, img_bin = cv2.threshold(img_r, 127, 255, cv2.THRESH_BINARY)
        
        # Slide overlay
        for x_offset in range(0, target.shape[1] - w + 1, 10):
            target_sub = target_bin[:, x_offset:x_offset+w]
            match = np.sum(img_bin == target_sub)
            sim = match / (h * w)
            if sim > max_sim:
                max_sim = sim
                best_w = w
                best_x = x_offset
    print(f"File: {f} | Max Similarity: {max_sim:.4f} | width: {best_w} | offset: {best_x}")
