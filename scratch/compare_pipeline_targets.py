import cv2
import numpy as np

# Load pipeline binarised image
pipeline_img = cv2.imread("data/binarised/tamil_stone__tamil_026_jpg_binarised.png", cv2.IMREAD_GRAYSCALE)
pipeline_inv = cv2.bitwise_not(pipeline_img)

targets = {
    "media__1782218721649.png": r"C:\Users\nanda_4h6zihz\.gemini\antigravity-ide\brain\9bb206b2-f39c-4a1f-befa-83279631baab\media__1782218721649.png",
    "media__1782219053166.png": r"C:\Users\nanda_4h6zihz\.gemini\antigravity-ide\brain\9bb206b2-f39c-4a1f-befa-83279631baab\media__1782219053166.png"
}

for name, path in targets.items():
    t_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if t_img is None:
        continue
    _, t_bin = cv2.threshold(t_img, 127, 255, cv2.THRESH_BINARY)
    H_t, W_t = t_img.shape
    
    # Test resizing pipeline_inv to match the target
    for w in [509, 531, 581, 632]:
        h = H_t
        p_inv_r = cv2.resize(pipeline_inv, (w, h), interpolation=cv2.INTER_NEAREST)
        _, p_inv_r_bin = cv2.threshold(p_inv_r, 127, 255, cv2.THRESH_BINARY)
        
        # Slide overlay
        max_sim = 0
        for x_offset in range(0, W_t - w + 1, 10):
            t_sub = t_bin[:, x_offset:x_offset+w]
            match = np.sum(p_inv_r_bin == t_sub)
            sim = match / t_sub.size
            if sim > max_sim:
                max_sim = sim
        print(f"Target: {name} | Width: {w} | Max similarity: {max_sim:.4f}")
