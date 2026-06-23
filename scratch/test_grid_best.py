import cv2
import numpy as np
from skimage.filters import threshold_sauvola

img_path = "data/binarised_representative_samples/tamil_stone/tamil_026_original.jpg"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply grid search best configuration
# 1. Gaussian blur (k=5)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 2. Sauvola ws=61, k=0.3
thresh = threshold_sauvola(blurred, window_size=61, k=0.3)
binary = (blurred < thresh).astype(np.uint8) * 255

# 3. Morph close (2x2)
binary_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

# 4. Invert to black-on-white
binary_inv = cv2.bitwise_not(binary_close)

# Save output
out_path = "scratch/grid_best_inv.png"
cv2.imwrite(out_path, binary_inv)
print(f"Saved best grid-search configuration to {out_path}")

# Calculate stats
total = binary_inv.size
black = np.sum(binary_inv == 0)
white = np.sum(binary_inv == 255)
print(f"Grid Best: black (text)={black} ({black/total*100:.2f}%), white (bg)={white} ({white/total*100:.2f}%)")
