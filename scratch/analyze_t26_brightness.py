import cv2
import numpy as np

img = cv2.imread("data/binarised_representative_samples/tamil_stone/tamil_026_original.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
H, W = gray.shape

print(f"tamil_026_original: shape={gray.shape}")
print(f"Mean brightness: {gray.mean():.2f}")
print(f"Std dev of brightness: {gray.std():.2f}")
print(f"Min brightness: {gray.min()}")
print(f"Max brightness: {gray.max()}")

# Save a simple adaptive threshold of the original image
# both polarities (dark on light vs light on dark)
_, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
_, otsu_dir = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imwrite("scratch/otsu_inv.png", otsu_inv)
cv2.imwrite("scratch/otsu_dir.png", otsu_dir)
print(f"Otsu Inv (white text if dark-on-light) mean: {otsu_inv.mean():.2f}")
print(f"Otsu Dir (white text if light-on-dark) mean: {otsu_dir.mean():.2f}")
