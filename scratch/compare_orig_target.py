import cv2
import numpy as np

# Load target
target = cv2.imread(r"C:\Users\nanda_4h6zihz\.gemini\antigravity-ide\brain\9bb206b2-f39c-4a1f-befa-83279631baab\media__1782219053166.png", cv2.IMREAD_GRAYSCALE)
H_t, W_t = target.shape

# Load original
orig = cv2.imread("data/binarised_representative_samples/tamil_stone/tamil_026_original.jpg", cv2.IMREAD_GRAYSCALE)

# Resize orig to target size
orig_r = cv2.resize(orig, (W_t, H_t), interpolation=cv2.INTER_LINEAR)

# Compute absolute difference and correlation
diff = cv2.absdiff(orig_r, target)
mean_diff = diff.mean()
correlation = np.corrcoef(orig_r.flat, target.flat)[0, 1]

print(f"Mean pixel difference: {mean_diff:.2f}")
print(f"Correlation: {correlation:.4f}")
