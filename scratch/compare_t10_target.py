import cv2
import numpy as np

# Load target
target = cv2.imread(r"C:\Users\nanda_4h6zihz\.gemini\antigravity-ide\brain\9bb206b2-f39c-4a1f-befa-83279631baab\media__1782219053166.png", cv2.IMREAD_GRAYSCALE)
H_t, W_t = target.shape

# Let's test with:
# 1. tamil_010_original.jpg (resized)
# 2. tamil_010_binarised_FIXED.png (resized, inverted or normal)
t10_orig = cv2.imread("data/binarised_representative_samples/tamil_stone/tamil_010_original.jpg", cv2.IMREAD_GRAYSCALE)
t10_fixed = cv2.imread("data/binarised_representative_samples/tamil_stone/tamil_010_binarised_FIXED.png", cv2.IMREAD_GRAYSCALE)

if t10_orig is not None:
    t10_orig_r = cv2.resize(t10_orig, (W_t, H_t), interpolation=cv2.INTER_LINEAR)
    corr = np.corrcoef(t10_orig_r.flat, target.flat)[0, 1]
    print(f"tamil_010_orig | Correlation: {corr:.4f}")

if t10_fixed is not None:
    t10_fixed_r = cv2.resize(t10_fixed, (W_t, H_t), interpolation=cv2.INTER_NEAREST)
    corr_normal = np.corrcoef(t10_fixed_r.flat, target.flat)[0, 1]
    corr_inv = np.corrcoef(cv2.bitwise_not(t10_fixed_r).flat, target.flat)[0, 1]
    print(f"tamil_010_fixed (normal) | Correlation: {corr_normal:.4f}")
    print(f"tamil_010_fixed (inverted) | Correlation: {corr_inv:.4f}")
