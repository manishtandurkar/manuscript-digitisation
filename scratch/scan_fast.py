import cv2
import numpy as np
from pathlib import Path

target = cv2.imread(r"C:\Users\nanda_4h6zihz\.gemini\antigravity-ide\brain\9bb206b2-f39c-4a1f-befa-83279631baab\media__1782219053166.png", cv2.IMREAD_GRAYSCALE)
H_t, W_t = target.shape

results = []

# Paths to scan
paths = [
    Path("data/binarised_representative_samples"),
    Path("data/binarised")
]

for p_dir in paths:
    if not p_dir.exists():
        continue
    for p in p_dir.glob("**/*"):
        if p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif"]:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            for inverted in [False, True]:
                test_img = cv2.bitwise_not(img) if inverted else img
                test_r = cv2.resize(test_img, (W_t, H_t), interpolation=cv2.INTER_LINEAR)
                corr = np.corrcoef(test_r.flat, target.flat)[0, 1]
                if not np.isnan(corr):
                    results.append((corr, p, inverted))

results.sort(key=lambda x: x[0], reverse=True)
print("Top 10 fast matches:")
for corr, p, inv in results[:10]:
    print(f"  Correlation: {corr:.4f} | File: {p} | Inverted: {inv}")
