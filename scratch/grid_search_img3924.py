import cv2
import numpy as np
from skimage.filters import threshold_sauvola

img_path = "data/raw/tamil_stone/IMG_3924.jpg"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
H, W = gray.shape

# Crop central region
cy, cx = H // 2, W // 2
crop = gray[cy-400:cy+400, cx-400:cx+400]

# Test different median blur sizes and Sauvola parameters
for ksize in [11, 21, 31]:
    blurred = cv2.medianBlur(crop, ksize)
    
    for ws in [51, 101, 151]:
        for k in [0.15, 0.25, 0.35]:
            thresh = threshold_sauvola(blurred, window_size=ws, k=k)
            binary = (blurred < thresh).astype(np.uint8) * 255
            
            # Count components and white pixels
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            white_pct = (binary == 255).mean() * 100
            
            # Save some promising ones
            if 3.0 < white_pct < 20.0 and 50 < num_labels < 1000:
                cv2.imwrite(f"scratch/img3924_crop_m{ksize}_ws{ws}_k{int(k*100)}_inv.png", cv2.bitwise_not(binary))
                print(f"SAVED: median={ksize} | ws={ws} | k={k:.2f} | white={white_pct:.2f}% | components={num_labels}")
            else:
                pass
