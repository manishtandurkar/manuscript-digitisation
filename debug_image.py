import cv2
import numpy as np
import hashlib

path = r"C:\6th semester EL's\Interdisciplinary project\Implementation\manuscript-digitisation\data\binarised_representative_samples\malayalam_stone\image15_original.jpeg"

with open(path, "rb") as f:
    raw_bytes = f.read()
print("File size (bytes):", len(raw_bytes))
print("MD5:", hashlib.md5(raw_bytes).hexdigest())

img = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
print("Shape:", img.shape)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("Gray mean/std:", gray.mean(), gray.std())

_, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
actual_t = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
print("Otsu computed threshold value:", actual_t)
print("Ink % at that threshold:", 100 * np.count_nonzero(otsu_thresh) / otsu_thresh.size)

cv2.imwrite("debug_raw_otsu_yourmachine.png", otsu_thresh)
print("Saved debug_raw_otsu_yourmachine.png next to wherever you ran this script")