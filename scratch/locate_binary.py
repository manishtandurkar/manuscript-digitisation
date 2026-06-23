import cv2
import numpy as np

# Load target
img = cv2.imread(r"C:\Users\nanda_4h6zihz\.gemini\antigravity-ide\brain\9bb206b2-f39c-4a1f-befa-83279631baab\media__1782219053166.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
H, W = gray.shape

# A binarised image contains mostly 0 and 255.
# Let's find local windows of size (300, 400) or similar that have very few intermediate gray values (between 10 and 245).
# Let's compute a map of "is_binary": 1 if pixel is < 10 or > 245, else 0.
is_binary = np.logical_or(gray < 10, gray > 245).astype(np.float32)

# Let's find the largest rectangular region in is_binary that is almost entirely 1.
# We can do this by computing a 2D box filter (average) of various sizes.
# Let's test sizes from 200x200 up to H x W.
best_val = 0
best_box = None

for h_w in range(200, H + 1, 10):
    for w_w in range(200, W + 1, 10):
        # We can use cv2.boxFilter or just manual scan if fast
        kernel = np.ones((h_w, w_w), np.float32) / (h_w * w_w)
        filtered = cv2.filter2D(is_binary, -1, kernel)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(filtered)
        if max_val > 0.95:  # 95% of pixels are binary
            # We want to find the largest one, so let's weigh by area
            score = max_val * (h_w * w_w)
            if score > best_val:
                best_val = score
                # max_loc is (x, y) of the top-left corner of the kernel
                # wait, filter2D output anchor is at center by default, so we need to adjust
                best_box = (max_loc[1] - h_w//2, max_loc[0] - w_w//2, h_w, w_w, max_val)

if best_box:
    y, x, h, w, val = best_box
    # Ensure indices are within bounds
    y = max(0, y)
    x = max(0, x)
    h = min(h, H - y)
    w = min(w, W - x)
    print(f"Best binary region found: y={y}, x={x}, h={h}, w={w}, accuracy={val:.4f}")
    cropped = img[y:y+h, x:x+w]
    cv2.imwrite("scratch/binary_target_cropped.png", cropped)
else:
    print("No binary region found")
