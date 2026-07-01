import cv2
import numpy as np

for name in ["a", "b"]:
    path = f"scratch/test_method_{name}_inv.png"
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        total = img.size
        black = np.sum(img == 0)
        white = np.sum(img == 255)
        print(f"Method {name.upper()}: black (text)={black} ({black/total*100:.2f}%), white (bg)={white} ({white/total*100:.2f}%)")
    else:
        print(f"Could not load method {name}")
