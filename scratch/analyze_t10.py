import cv2
import numpy as np

t10_orig = cv2.imread("data/binarised_representative_samples/tamil_stone/tamil_010_original.jpg")
t10_fixed = cv2.imread("data/binarised_representative_samples/tamil_stone/tamil_010_binarised_FIXED.png")

if t10_orig is not None:
    print(f"tamil_010_orig: shape={t10_orig.shape}")
if t10_fixed is not None:
    print(f"tamil_010_fixed: shape={t10_fixed.shape}, black={np.sum(t10_fixed == 0)}, white={np.sum(t10_fixed == 255)}")
