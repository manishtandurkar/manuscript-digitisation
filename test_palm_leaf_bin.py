import cv2
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from src.binarise import _palm_leaf_rough_mask, binarise_palm_leaf, remove_noise_blobs

def main():
    img_path = r"C:\6th semester EL's\Interdisciplinary project\Implementation\manuscript-digitisation\data\raw\Vijay Kumar extra images\img334.jpg"
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("Failed to load")
        return
        
    rough = _palm_leaf_rough_mask(img)
    print("Rough mean:", rough.mean())
    cv2.imwrite("tune_img334_out/rough_mask.png", rough)
    
    H, W = img.shape[:2]
    shorter = min(H, W)
    dil_k = max(3, shorter // 40)
    dilated = cv2.dilate(rough, np.ones((dil_k, dil_k), np.uint8))
    print("Dilated mean:", dilated.mean())
    cv2.imwrite("tune_img334_out/dilated_mask.png", dilated)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    print("Num labels:", num_labels)
    
    min_area = max(20, (shorter // 60) ** 2)
    max_area = int(H * W * 0.40)
    print("Min area:", min_area, "Max area:", max_area)
    
    valid_labels = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area and area <= max_area:
            valid_labels += 1
            
    print("Valid labels:", valid_labels)
    
    canvas = binarise_palm_leaf(img)
    print("Canvas mean:", canvas.mean())
    cv2.imwrite("tune_img334_out/canvas.png", canvas)
    
    cleaned = remove_noise_blobs(canvas, min_size=8, min_length=15)
    print("Cleaned mean:", cleaned.mean())
    cv2.imwrite("tune_img334_out/cleaned.png", cleaned)

if __name__ == "__main__":
    main()
