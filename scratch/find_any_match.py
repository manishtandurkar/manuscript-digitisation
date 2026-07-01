import cv2
import numpy as np
from skimage.filters import threshold_sauvola

# Load target
target = cv2.imread(r"C:\Users\nanda_4h6zihz\.gemini\antigravity-ide\brain\9bb206b2-f39c-4a1f-befa-83279631baab\media__1782219053166.png", cv2.IMREAD_GRAYSCALE)
_, target_bin = cv2.threshold(target, 127, 255, cv2.THRESH_BINARY)
H_t, W_t = target.shape

# Load original
orig = cv2.imread("data/binarised_representative_samples/tamil_stone/tamil_026_original.jpg", cv2.IMREAD_GRAYSCALE)

# We want to check:
# 1. Sauvola (ws, k)
# 2. Otsu
# 3. Adaptive Mean
# 4. Adaptive Gaussian
# 5. Dilation / Erosion / Closing / Opening on the result
# And check similarity by resizing to target shape.

best_sim = 0
best_info = ""
best_img = None

# Methods
methods = []

# 1. Sauvola
for ws in [9, 15, 21, 31, 41, 51, 61]:
    for k in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        thresh = threshold_sauvola(orig, window_size=ws, k=k)
        bin_sau = (orig < thresh).astype(np.uint8) * 255
        methods.append((f"sauvola_ws{ws}_k{k}", bin_sau))

# 2. Otsu
_, bin_otsu = cv2.threshold(orig, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
methods.append(("otsu", bin_otsu))

# 3. Adaptive Mean
for block in [5, 11, 15, 21, 31]:
    for C in [2, 5, 10, 15, 20]:
        bin_ad_m = cv2.adaptiveThreshold(orig, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, C)
        methods.append((f"adaptive_mean_b{block}_c{C}", bin_ad_m))

# 4. Adaptive Gaussian
for block in [5, 11, 15, 21, 31]:
    for C in [2, 5, 10, 15, 20]:
        bin_ad_g = cv2.adaptiveThreshold(orig, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, C)
        methods.append((f"adaptive_gauss_b{block}_c{C}", bin_ad_g))

print(f"Testing {len(methods)} base configurations...")

for name, base_bin in methods:
    # We will test various morphological operations on base_bin
    for morph in ["none", "close1", "close2", "close3", "open1", "open2"]:
        if morph == "none":
            m_bin = base_bin
        elif morph == "close1":
            m_bin = cv2.morphologyEx(base_bin, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        elif morph == "close2":
            m_bin = cv2.morphologyEx(base_bin, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        elif morph == "close3":
            m_bin = cv2.morphologyEx(base_bin, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        elif morph == "open1":
            m_bin = cv2.morphologyEx(base_bin, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        elif morph == "open2":
            m_bin = cv2.morphologyEx(base_bin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            
        # Try both polarities
        for polarity in [True, False]:
            src = m_bin if polarity else cv2.bitwise_not(m_bin)
            
            # Resize to target
            src_r = cv2.resize(src, (W_t, H_t), interpolation=cv2.INTER_NEAREST)
            _, src_bin = cv2.threshold(src_r, 127, 255, cv2.THRESH_BINARY)
            
            match = np.sum(src_bin == target_bin)
            sim = match / target_bin.size
            if sim > best_sim:
                best_sim = sim
                best_info = f"Method: {name} | Morph: {morph} | Polarity: {polarity} | Sim: {sim:.4f}"
                best_img = src.copy()
                print(f"New Best: {best_info}")

if best_img is not None:
    cv2.imwrite("scratch/best_search_match.png", best_img)
    cv2.imwrite("scratch/best_search_match_inv.png", cv2.bitwise_not(best_img))
    print("Done. Saved best match to scratch/best_search_match.png")
