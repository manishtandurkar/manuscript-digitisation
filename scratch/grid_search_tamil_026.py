import cv2
import numpy as np
import pathlib
from skimage.filters import threshold_sauvola

# Load target screenshot
target_path = r"C:\Users\nanda_4h6zihz\.gemini\antigravity-ide\brain\9bb206b2-f39c-4a1f-befa-83279631baab\media__1782218721649.png"
target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
target_crop = target[6:360, 4:506]
H_tc, W_tc = target_crop.shape[:2]
_, target_bin = cv2.threshold(target_crop, 127, 255, cv2.THRESH_BINARY)

# Load original image
img_path = 'data/binarised_representative_samples/tamil_stone/tamil_026_original.jpg'
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

best_sim = 0.0
best_params = None

# Let's test a very wide grid
window_sizes = [11, 15, 21, 25, 31, 41, 51, 61, 71]
k_values = [0.01, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]
blurs = [None, 3, 5]
closes = [0, 2, 3]

results = []

for ws in window_sizes:
    for k in k_values:
        for b in blurs:
            for c in closes:
                # 1. Blur
                if b is not None:
                    processed = cv2.GaussianBlur(gray, (b, b), 0)
                else:
                    processed = gray.copy()
                
                # 2. Sauvola
                thresh = threshold_sauvola(processed, window_size=ws, k=k)
                binary = (processed < thresh).astype(np.uint8) * 255
                
                # 3. Morph close
                if c > 0:
                    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((c, c), np.uint8))
                
                # Resize to target crop
                binary_r = cv2.resize(binary, (W_tc, H_tc), interpolation=cv2.INTER_NEAREST)
                _, binary_bin = cv2.threshold(binary_r, 127, 255, cv2.THRESH_BINARY)
                
                # Check both polarities
                matching = np.sum(binary_bin == target_bin)
                matching_inv = np.sum(cv2.bitwise_not(binary_bin) == target_bin)
                best_match = max(matching, matching_inv)
                sim = best_match / (H_tc * W_tc)
                
                if sim > best_sim:
                    best_sim = sim
                    best_params = (ws, k, b, c, matching_inv > matching)
                    print(f"New Best Similarity: {sim:.4f} with ws={ws}, k={k}, blur={b}, close={c}, inverted={matching_inv > matching}")
                
                results.append((sim, ws, k, b, c, matching_inv > matching))

results.sort(key=lambda x: x[0], reverse=True)
print("\nTop 10 parameters:")
for sim, ws, k, b, c, inv in results[:10]:
    print(f"  Similarity: {sim:.4f} | ws={ws}, k={k}, blur={b}, close={c}, inv={inv}")
