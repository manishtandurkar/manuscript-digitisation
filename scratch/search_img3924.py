import os
from pathlib import Path
import cv2

for root, dirs, files in os.walk("data"):
    for f in files:
        if "IMG_3924" in f:
            p = Path(os.path.join(root, f))
            try:
                img = cv2.imread(str(p))
                shape_str = str(img.shape) if img is not None else "Cannot read"
            except Exception as e:
                shape_str = f"Error: {e}"
            print(f"File: {p} | Size: {p.stat().st_size} bytes | Shape: {shape_str}")
