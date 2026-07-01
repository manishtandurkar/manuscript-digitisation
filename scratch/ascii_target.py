import cv2
import numpy as np

img = cv2.imread(r"C:\Users\nanda_4h6zihz\.gemini\antigravity-ide\brain\9bb206b2-f39c-4a1f-befa-83279631baab\media__1782219053166.png", cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Could not load image.")
    exit(1)

# Resize to 40 columns and 20 rows
rows, cols = 20, 60
resized = cv2.resize(img, (cols, rows), interpolation=cv2.INTER_AREA)

# Map grayscale values to ASCII characters
chars = " .:-=+*#%@" # light to dark (since we want to print on dark/light terminal)
# Or simple block characters
# Let's map 0-255 to chars
print("--- Text representation of media__1782219053166.png ---")
for r in range(rows):
    line = ""
    for c in range(cols):
        val = resized[r, c]
        char_idx = int(val / 256.0 * len(chars))
        line += chars[char_idx]
    print(line)
print("-----------------------------------------------------")
