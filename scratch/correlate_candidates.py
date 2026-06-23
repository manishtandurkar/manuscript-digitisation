import cv2
import numpy as np

# Load target
target = cv2.imread(r"C:\Users\nanda_4h6zihz\.gemini\antigravity-ide\brain\9bb206b2-f39c-4a1f-befa-83279631baab\media__1782219053166.png", cv2.IMREAD_GRAYSCALE)
H_t, W_t = target.shape

candidates = {
    "Method A (binarise_stone default)": "scratch/test_method_a_inv.png",
    "Method B (sauvola ws=15, k=0.25)": "scratch/test_method_b_inv.png",
    "Grid Best (ws=61, k=0.3, blur=5)": "scratch/grid_best_inv.png",
    "Pipeline (binarised.png inverted)": "data/binarised/tamil_stone__tamil_026_jpg_binarised.png"
}

for name, path in candidates.items():
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    # If it is the pipeline image, invert it first
    if "Pipeline" in name:
        img = cv2.bitwise_not(img)
        
    # Resize to target
    img_r = cv2.resize(img, (W_t, H_t), interpolation=cv2.INTER_LINEAR)
    
    # Compute correlation
    corr = np.corrcoef(img_r.flat, target.flat)[0, 1]
    print(f"Candidate: {name} | Correlation: {corr:.4f}")
