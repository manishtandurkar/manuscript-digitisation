import cv2
import numpy as np
from skimage.filters import threshold_sauvola

img_path = r"data/binarised_representative_samples/tamil_stone/tamil_026_original.jpg"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Sauvola ws=15, k=0.05
thresh1 = threshold_sauvola(gray, window_size=15, k=0.05)
bin1 = (gray < thresh1).astype(np.uint8) * 255
cv2.imwrite("scratch/raw_sauvola_k005_ws15.png", bin1)
cv2.imwrite("scratch/raw_sauvola_k005_ws15_inv.png", cv2.bitwise_not(bin1))

# Sauvola ws=61, k=0.3
thresh2 = threshold_sauvola(gray, window_size=61, k=0.3)
bin2 = (gray < thresh2).astype(np.uint8) * 255
cv2.imwrite("scratch/raw_sauvola_k03_ws61.png", bin2)
cv2.imwrite("scratch/raw_sauvola_k03_ws61_inv.png", cv2.bitwise_not(bin2))

# Otsu
_, bin_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imwrite("scratch/raw_otsu.png", bin_otsu)
cv2.imwrite("scratch/raw_otsu_inv.png", cv2.bitwise_not(bin_otsu))

print("Saved raw binarisations.")
