import cv2
import numpy as np

img = cv2.imread(r"C:\Users\nanda_4h6zihz\.gemini\antigravity-ide\brain\9bb206b2-f39c-4a1f-befa-83279631baab\media__1782219053166.png", cv2.IMREAD_GRAYSCALE)
H, W = img.shape
mid = W // 2

# Left half
left = img[:, :mid]
left_binary_pct = np.logical_or(left < 10, left > 245).mean() * 100

# Right half
right = img[:, mid:]
right_binary_pct = np.logical_or(right < 10, right > 245).mean() * 100

print(f"Image shape: {img.shape}")
print(f"Left half binary pixel %: {left_binary_pct:.2f}%")
print(f"Right half binary pixel %: {right_binary_pct:.2f}%")

# Let's also check column by column binary percentage to find the exact boundary
col_binary_pcts = [np.logical_or(img[:, c] < 10, img[:, c] > 245).mean() * 100 for c in range(W)]
# print some key columns
print(f"Col binary % (every 30 cols): {[f'{c}:{col_binary_pcts[c]:.1f}%' for c in range(0, W, 30)]}")
