import cv2
import numpy as np

orig = cv2.imread("data/binarised_representative_samples/tamil_stone/tamil_026_original.jpg", cv2.IMREAD_GRAYSCALE)
target = cv2.imread(r"C:\Users\nanda_4h6zihz\.gemini\antigravity-ide\brain\9bb206b2-f39c-4a1f-befa-83279631baab\media__1782219053166.png", cv2.IMREAD_GRAYSCALE)

print(f"Original image shape: {orig.shape}")
print(f"Original left column (first 5px average): {orig[:, 0:5].mean():.2f}")
print(f"Original right column (last 5px average): {orig[:, -5:].mean():.2f}")

print(f"Target image shape: {target.shape}")
print(f"Target left column (first 5px average): {target[:, 0:5].mean():.2f}")
print(f"Target right column (last 5px average): {target[:, -5:].mean():.2f}")
