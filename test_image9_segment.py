import cv2
import numpy as np
from pathlib import Path
from skimage.filters import threshold_sauvola

def main():
    img_path = r"C:\6th semester EL's\Interdisciplinary project\Implementation\manuscript-digitisation\data\raw\malayalam_stone\image9.png"
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("Failed to load")
        return

    H, W = img.shape[:2]
    shorter = min(H, W)
    
    # 1. Convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("tune_img334_out/image9_gray.png", gray)

    # 2. Segment the plate: background is white (> 200), plate is dark (< 150)
    # Let's create a plate mask
    _, plate_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    # Clean up plate mask (remove tiny specs, keep the large plate rectangle)
    kernel_size = max(5, shorter // 30)
    plate_mask = cv2.morphologyEx(plate_mask, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8))
    plate_mask = cv2.morphologyEx(plate_mask, cv2.MORPH_OPEN, np.ones((kernel_size, kernel_size), np.uint8))
    cv2.imwrite("tune_img334_out/image9_plate_mask.png", plate_mask)

    # Apply bilateral filter to smooth text texture inside the plate
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=30, sigmaSpace=30)

    # 3. Create rough text mask: text is LIGHT, background is DARK
    # Since text is light, we look for pixels that are significantly brighter than local mean.
    # In Sauvola, thresh = mean * (1 + k * (std / R - 1)).
    # Bright text means: denoised > thresh
    ws = 31
    k = 0.12
    thresh = threshold_sauvola(denoised, window_size=ws, k=k)
    rough_text = (denoised > thresh).astype(np.uint8) * 255
    # Apply plate mask to restrict to the copper plate area
    rough_text[plate_mask == 0] = 0
    cv2.imwrite("tune_img334_out/image9_rough_text.png", rough_text)

    # 4. Character-level segmentation (extracting each letter)
    # Dilate rough text to group individual characters
    dil_k = 3
    dilated = cv2.dilate(rough_text, np.ones((dil_k, dil_k), np.uint8))
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    
    min_area = 15
    max_area = int(H * W * 0.05) # characters are small
    
    canvas = np.zeros((H, W), dtype=np.uint8)
    pad = 2
    
    print(f"Total components found: {num_labels - 1}")
    valid_count = 0

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
            
        x  = int(stats[label, cv2.CC_STAT_LEFT])
        y  = int(stats[label, cv2.CC_STAT_TOP])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        
        # Crop bounds padded
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(W, x + cw + pad)
        y1 = min(H, y + ch + pad)
        
        crop_gray = denoised[y0:y1, x0:x1]
        if crop_gray.size == 0:
            continue
            
        # Local threshold on crop
        cr_h, cr_w = crop_gray.shape
        if cr_h < 5 or cr_w < 5:
            _, local_bin = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            local_ws = max(5, min(15, min(cr_h, cr_w) // 2 | 1))
            local_thresh = threshold_sauvola(crop_gray, window_size=local_ws, k=0.08)
            local_bin = (crop_gray > local_thresh).astype(np.uint8) * 255
            
        # Stamp back
        canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], local_bin)
        valid_count += 1
        
    print(f"Stamped valid components: {valid_count}")
    
    # Mask final output to copper plate
    canvas[plate_mask == 0] = 0
    
    # Post-processing clean-up
    canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    
    # Remove remaining small isolated noise blobs
    # Filter using connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(canvas, connectivity=8)
    cleaned = np.zeros_like(canvas)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area >= 10: # keep small components
            cleaned[labels == label] = 255

    cv2.imwrite("tune_img334_out/image9_segmented.png", cleaned)
    print("Saved tune_img334_out/image9_segmented.png")

if __name__ == "__main__":
    main()
