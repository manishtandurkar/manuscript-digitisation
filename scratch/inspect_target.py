import cv2
import numpy as np

img = cv2.imread(r"C:\Users\nanda_4h6zihz\.gemini\antigravity-ide\brain\9bb206b2-f39c-4a1f-befa-83279631baab\media__1782219053166.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Let's save a grayscale version of the image so we can inspect it, or write text stats.
# Find large bounding boxes of uniform color (like borders or containers).
# Since it is a screenshot, let's find the bounding box of the main content area.
# Usually, the main content area has a white or black background.
print(f"Image shape: {img.shape}")
print(f"Unique gray values and their counts (top 10):")
vals, counts = np.unique(gray, return_counts=True)
idx = np.argsort(-counts)
for i in range(min(10, len(idx))):
    print(f"  value: {vals[idx[i]]}, count: {counts[idx[i]]} ({counts[idx[i]]/gray.size*100:.2f}%)")

# Let's check the bounding box where there is actual content
# Binarize with Otsu to separate background from foreground UI elements.
_, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
coords = np.column_stack(np.where(thresh > 0))
if len(coords) > 0:
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    print(f"Content bounding box: y=[{y_min}, {y_max}], x=[{x_min}, {x_max}]")
    cropped = img[y_min:y_max+1, x_min:x_max+1]
    cv2.imwrite(r"C:\Users\nanda_4h6zihz\.gemini\antigravity-ide\brain\9bb206b2-f39c-4a1f-befa-83279631baab\media_target_cropped.png", cropped)
    print("Saved cropped target to media_target_cropped.png")
