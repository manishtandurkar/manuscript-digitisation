import cv2
import numpy as np
import shutil
from pathlib import Path

# Load full-image binarised output
full_bin = cv2.imread("scratch/IMG_3924_binarised_segmental.png", cv2.IMREAD_GRAYSCALE)
H, W = full_bin.shape

# Crop same region
cy, cx = H // 2, W // 2
crop_integrated = full_bin[cy-400:cy+400, cx-400:cx+400]

# Save crop
out_crop_path = "scratch/img3924_crop_final_integrated.png"
cv2.imwrite(out_crop_path, crop_integrated)

# Check against Option C stand-alone (which was saved inverted, so we invert crop_integrated to compare)
crop_integrated_inv = cv2.bitwise_not(crop_integrated)
option_c = cv2.imread("scratch/img3924_crop_m31_ws101_k25_inv.png", cv2.IMREAD_GRAYSCALE)

if option_c is not None:
    diff = cv2.absdiff(crop_integrated_inv, option_c)
    non_zero = np.count_nonzero(diff)
    print(f"Difference pixels between full-run crop and standalone Option C: {non_zero} (out of {diff.size})")
    
# Copy to brain directory for user verification
brain_dir = Path(r"C:\Users\nanda_4h6zihz\.gemini\antigravity-ide\brain\9bb206b2-f39c-4a1f-befa-83279631baab")
shutil.copy(out_crop_path, str(brain_dir / "img3924_crop_final_integrated.png"))
print("Copied final integrated crop to brain directory.")
