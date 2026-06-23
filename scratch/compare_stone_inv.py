import cv2
import numpy as np
from src.binarise import binarise_stone

img_path = "data/binarised_representative_samples/tamil_stone/tamil_026_original.jpg"
img = cv2.imread(img_path)
bin_stone = binarise_stone(img)

# Invert to black-on-white
bin_stone_inv = cv2.bitwise_not(bin_stone)

# Save the output
out_path = "data/binarised_representative_samples/tamil_stone/tamil_026_binarised_FIXED.png"
cv2.imwrite(out_path, bin_stone_inv)
print(f"Saved inverted binarise_stone output to {out_path}")

# Load target to compare similarity
target = cv2.imread(r"C:\Users\nanda_4h6zihz\.gemini\antigravity-ide\brain\9bb206b2-f39c-4a1f-befa-83279631baab\media__1782219053166.png", cv2.IMREAD_GRAYSCALE)
_, target_bin = cv2.threshold(target, 127, 255, cv2.THRESH_BINARY)
H_t, W_t = target.shape

# Resize our inverted output to target size and compute similarity
bin_stone_inv_r = cv2.resize(bin_stone_inv, (W_t, H_t), interpolation=cv2.INTER_NEAREST)
_, bin_stone_inv_r_bin = cv2.threshold(bin_stone_inv_r, 127, 255, cv2.THRESH_BINARY)

match = np.sum(bin_stone_inv_r_bin == target_bin)
sim = match / target_bin.size
print(f"Similarity of inverted binarise_stone output to target screenshot: {sim:.4f}")
