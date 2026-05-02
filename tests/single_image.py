import cv2
import numpy as np
from skimage.filters import threshold_sauvola, threshold_niblack
from pathlib import Path

img_path = r"C:\6th semester EL's\Interdisciplinary project\Implementation\manuscript-digitisation\data\raw\malayalam_stone\image1.jpeg"  # change this
img = cv2.imread(img_path)
out = Path("palm_debug_outputs")
out.mkdir(exist_ok=True)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 1. raw grayscale
cv2.imwrite(str(out/"00_gray.png"), gray)

# 2. invert (sometimes ink is lighter than bg in grayscale)
cv2.imwrite(str(out/"01_inverted.png"), cv2.bitwise_not(gray))

# 3. sauvola small window
t = threshold_sauvola(gray, window_size=25, k=0.1)
cv2.imwrite(str(out/"02_sauvola_25_k01.png"), ((gray < t)*255).astype(np.uint8))

# 4. sauvola large window  
t = threshold_sauvola(gray, window_size=75, k=0.1)
cv2.imwrite(str(out/"03_sauvola_75_k01.png"), ((gray < t)*255).astype(np.uint8))

# 5. sauvola on inverted
t = threshold_sauvola(cv2.bitwise_not(gray), window_size=51, k=0.1)
cv2.imwrite(str(out/"04_sauvola_inverted.png"), ((cv2.bitwise_not(gray) < t)*255).astype(np.uint8))

# 6. otsu on gray
_, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite(str(out/"05_otsu.png"), b)

# 7. otsu inverted
_, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imwrite(str(out/"06_otsu_inv.png"), b)

# 8. adaptive mean
b = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 5)
cv2.imwrite(str(out/"07_adaptive_mean_31.png"), b)

# 9. adaptive gaussian
b = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
cv2.imwrite(str(out/"08_adaptive_gaussian_31.png"), b)

# 10. on just the blue channel (sometimes separates ink better on palm leaf)
B, G, R = cv2.split(img)
t = threshold_sauvola(B, window_size=51, k=0.1)
cv2.imwrite(str(out/"09_sauvola_blue_ch.png"), ((B < t)*255).astype(np.uint8))

# 11. on green channel
t = threshold_sauvola(G, window_size=51, k=0.1)
cv2.imwrite(str(out/"10_sauvola_green_ch.png"), ((G < t)*255).astype(np.uint8))

# 12. on red channel
t = threshold_sauvola(R, window_size=51, k=0.1)
cv2.imwrite(str(out/"11_sauvola_red_ch.png"), ((R < t)*255).astype(np.uint8))

# 13. morphological tophat (reveals dark strokes on textured bg)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
tophat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
cv2.imwrite(str(out/"12_blackhat.png"), tophat)
_, b = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite(str(out/"13_blackhat_otsu.png"), b)

print("Done. Check the palm_debug_outputs folder.")
print(f"Gray stats — min:{gray.min()} max:{gray.max()} mean:{gray.mean():.1f}")
B,G,R = cv2.split(img)
print(f"B channel mean:{B.mean():.1f}  G:{G.mean():.1f}  R:{R.mean():.1f}")