import cv2
import numpy as np

p = "data/binarised/tamil_stone__tamil_026_jpg_binarised.png"
img = cv2.imread(p)
if img is not None:
    print(f"Binarised pipeline image: shape={img.shape}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    total = gray.size
    black = np.sum(gray == 0)
    white = np.sum(gray == 255)
    print(f"  black={black} ({black/total*100:.2f}%), white={white} ({white/total*100:.2f}%)")
else:
    print("Could not read image from pipeline")
