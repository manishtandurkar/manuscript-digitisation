import cv2
import numpy as np

# Load target
img = cv2.imread(r"C:\Users\nanda_4h6zihz\.gemini\antigravity-ide\brain\9bb206b2-f39c-4a1f-befa-83279631baab\media__1782219053166.png")
if img is None:
    print("Failed to load target image.")
    exit(1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
H, W = gray.shape

# Compute is_binary (1 if close to 0 or 255, else 0)
is_binary = np.logical_or(gray < 10, gray > 245).astype(np.uint8)

# Compute integral image of is_binary
integral = cv2.integral(is_binary)

best_score = 0
best_box = None

# Scan sizes
for h_w in range(200, H + 1, 10):
    for w_w in range(200, W + 1, 10):
        # We want to scan all possible positions (y, x)
        for y in range(0, H - h_w + 1, 10):
            for x in range(0, W - w_w + 1, 10):
                # Sum of rectangle using integral image:
                # S = I(y+h, x+w) - I(y+h, x) - I(y, x+w) + I(y, x)
                rect_sum = (integral[y + h_w, x + w_w] 
                            - integral[y + h_w, x] 
                            - integral[y, x + w_w] 
                            + integral[y, x])
                accuracy = rect_sum / (h_w * w_w)
                if accuracy > 0.95:
                    score = accuracy * (h_w * w_w)
                    if score > best_score:
                        best_score = score
                        best_box = (y, x, h_w, w_w, accuracy)

if best_box:
    y, x, h, w, acc = best_box
    print(f"Fast binary region found: y={y}, x={x}, h={h}, w={w}, accuracy={acc:.4f}")
    cropped = img[y:y+h, x:x+w]
    cv2.imwrite("scratch/binary_target_cropped.png", cropped)
    print("Saved cropped target to scratch/binary_target_cropped.png")
else:
    print("No binary region found")
