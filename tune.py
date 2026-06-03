import cv2
import numpy as np
from pathlib import Path

p = Path("data") / "raw" / "palm_leaf" / "image1.jpeg"
img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

out = Path("palm_tune")
out.mkdir(exist_ok=True)

for block in [21, 31, 41, 51]:
    for C in [3, 5, 8, 10]:
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block, C
        )
        h, w = binary.shape
        mask = np.zeros((h+2, w+2), np.uint8)
        for corner in [(0,0),(0,w-1),(h-1,0),(h-1,w-1)]:
            cv2.floodFill(binary, mask, (corner[1], corner[0]), 0)
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite(str(out / f"block{block}_C{C}.png"), binary)
        print(f"Saved block{block}_C{C}.png")

print("Done. Check palm_tune folder.")