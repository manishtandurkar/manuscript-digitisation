import cv2
import numpy as np
import os
from pathlib import Path

target = cv2.imread(r"C:\Users\nanda_4h6zihz\.gemini\antigravity-ide\brain\9bb206b2-f39c-4a1f-befa-83279631baab\media__1782219053166.png", cv2.IMREAD_GRAYSCALE)
H_t, W_t = target.shape

results = []

# Scan the data folder for images
for root, dirs, files in os.walk("data"):
    for f in files:
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            p = os.path.join(root, f)
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            # Try normal and inverted
            for inverted in [False, True]:
                test_img = cv2.bitwise_not(img) if inverted else img
                test_r = cv2.resize(test_img, (W_t, H_t), interpolation=cv2.INTER_LINEAR)
                corr = np.corrcoef(test_r.flat, target.flat)[0, 1]
                if not np.isnan(corr):
                    results.append((corr, p, inverted))

results.sort(key=lambda x: x[0], reverse=True)
print("Top 10 matching files in project data/ directory:")
for corr, p, inv in results[:10]:
    print(f"  Correlation: {corr:.4f} | File: {p} | Inverted: {inv}")
