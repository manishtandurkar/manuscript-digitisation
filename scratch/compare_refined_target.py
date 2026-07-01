import cv2
import numpy as np

target = cv2.imread(r"C:\Users\nanda_4h6zihz\.gemini\antigravity-ide\brain\9bb206b2-f39c-4a1f-befa-83279631baab\media__1782219053166.png", cv2.IMREAD_GRAYSCALE)
_, target_bin = cv2.threshold(target, 127, 255, cv2.THRESH_BINARY)
H_t, W_t = target.shape

img = cv2.imread("scratch/segmental_clean_output.png", cv2.IMREAD_GRAYSCALE)

for w in [509, 531, 581, 632]:
    img_r = cv2.resize(img, (w, H_t), interpolation=cv2.INTER_NEAREST)
    _, img_bin = cv2.threshold(img_r, 127, 255, cv2.THRESH_BINARY)
    
    max_sim = 0
    best_x = 0
    for x_offset in range(0, W_t - w + 1, 10):
        t_sub = target_bin[:, x_offset:x_offset+w]
        match = np.sum(img_bin == t_sub)
        sim = match / t_sub.size
        if sim > max_sim:
            max_sim = sim
            best_x = x_offset
    print(f"Width: {w} | Max similarity: {max_sim:.4f} | Offset: {best_x}")
