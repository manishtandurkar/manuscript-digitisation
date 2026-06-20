import cv2
import numpy as np
from pathlib import Path

img_path = "data/binarised_representative_samples/kannada_stone/image2_original.jpeg"
img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mean_sat = float(hsv[:,:,1].mean())
print(f"mean_sat: {mean_sat:.1f}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Local texture: std of a small local window, averaged
local_std = cv2.GaussianBlur((gray.astype(np.float32) - cv2.GaussianBlur(gray.astype(np.float32), (15,15), 0))**2, (15,15), 0)**0.5
print(f"mean local_std (speckle texture): {local_std.mean():.2f}")

print(f"global std: {gray.std():.2f}")