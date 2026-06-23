import cv2
import numpy as np
from skimage.filters import threshold_sauvola

img_path = r"data/binarised_representative_samples/tamil_stone/tamil_026_original.jpg"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

configs = [
    (15, 0.25),
    (25, 0.25),
    (35, 0.30),
    (51, 0.40),
]

for ws, k in configs:
    thresh = threshold_sauvola(gray, window_size=ws, k=k)
    binary = (gray < thresh).astype(np.uint8) * 255
    # Morph close
    binary_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    
    # Save inverted (black text on white background)
    cv2.imwrite(f"scratch/sauvola_ws{ws}_k{int(k*100)}_inv.png", cv2.bitwise_not(binary_close))
    print(f"Saved Sauvola ws={ws}, k={k:.2f} to scratch/sauvola_ws{ws}_k{int(k*100)}_inv.png")
