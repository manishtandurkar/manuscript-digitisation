import cv2
import numpy as np

path = r"C:\6th semester EL's\Interdisciplinary project\Implementation\manuscript-digitisation\data\binarised_representative_samples\malayalam_stone\image15_original.jpeg"

img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite("debug_step1_otsu.png", binary)
print("Step 1 (raw Otsu) ink%:", 100 * np.count_nonzero(binary) / binary.size)

h, w = gray.shape
min_size = max(4, (min(h, w) // 200) ** 2)
print("h,w =", h, w, " min_size =", min_size)

n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
print("Number of components found:", n - 1)

cleaned = np.zeros_like(binary)
kept = 0
for i in range(1, n):
    if stats[i, cv2.CC_STAT_AREA] >= min_size:
        cleaned[labels == i] = 255
        kept += 1
print("Components kept after min_size filter:", kept)
cv2.imwrite("debug_step2_cleaned.png", cleaned)
print("Step 2 (cleaned) ink%:", 100 * np.count_nonzero(cleaned) / cleaned.size)

print("cleaned.mean() =", cleaned.mean())
if cleaned.mean() >= 127:
    cleaned = cv2.bitwise_not(cleaned)
    print("Polarity was flipped!")
cv2.imwrite("debug_step3_final.png", cleaned)
print("Step 3 (final) ink%:", 100 * np.count_nonzero(cleaned) / cleaned.size)