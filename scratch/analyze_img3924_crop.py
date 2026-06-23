import cv2
import numpy as np
from skimage.filters import threshold_sauvola

img_path = "data/raw/tamil_stone/IMG_3924.jpg"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
H, W = gray.shape

# Let's crop a central region of 800x800 where characters are located
# According to the slider screenshot, characters are in the upper-middle area
cy, cx = H // 2, W // 2
crop = gray[cy-400:cy+400, cx-400:cx+400]
cv2.imwrite("scratch/img3924_crop_gray.jpg", crop)

# Let's test Sauvola binarisation on this crop
ws = 51
k = 0.15
thresh = threshold_sauvola(crop, window_size=ws, k=k)
binary = (crop < thresh).astype(np.uint8) * 255

cv2.imwrite("scratch/img3924_crop_sauvola_raw.png", binary)
cv2.imwrite("scratch/img3924_crop_sauvola_inv.png", cv2.bitwise_not(binary))

# Let's see what happens with morph close
binary_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
cv2.imwrite("scratch/img3924_crop_sauvola_close_inv.png", cv2.bitwise_not(binary_close))

# Let's check connected components sizes in the crop
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_close, connectivity=8)
areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
areas.sort(reverse=True)
print(f"Top 20 component areas in 800x800 crop: {areas[:20]}")
