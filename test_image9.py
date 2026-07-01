import cv2
import numpy as np
from pathlib import Path
from skimage.filters import threshold_sauvola

def main():
    # Exact path for Malayalam stone image9
    img_path = r"C:\6th semester EL's\Interdisciplinary project\Implementation\manuscript-digitisation\data\raw\malayalam_stone\image9.png"
    
    print(f"Reading image from: {img_path}")
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: Could not load image from {img_path}")
        return

    H, W = img.shape[:2]
    shorter = min(H, W)
    
    # 1. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Extract the rectangular copper plate mask
    # Background around the plate is bright white (> 180), plate is dark brown (< 180)
    _, plate_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological cleaning of plate mask to get a solid rectangle
    k_size = max(5, shorter // 30)
    plate_mask = cv2.morphologyEx(plate_mask, cv2.MORPH_CLOSE, np.ones((k_size, k_size), np.uint8))
    plate_mask = cv2.morphologyEx(plate_mask, cv2.MORPH_OPEN, np.ones((k_size, k_size), np.uint8))
    
    # Find bounding box of the copper plate
    contours, _ = cv2.findContours(plate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Error: Could not find plate boundary")
        return
        
    large_contour = max(contours, key=cv2.contourArea)
    px, py, pw, ph = cv2.boundingRect(large_contour)
    
    # Crop slightly inward (6px inset) to exclude the outer white border line
    border_inset = 6
    cx0 = max(0, px + border_inset)
    cy0 = max(0, py + border_inset)
    cx1 = min(W, px + pw - border_inset)
    cy1 = min(H, py + ph - border_inset)
    
    crop_gray = gray[cy0:cy1, cx0:cx1]
    
    # Bilateral filter to smooth texture inside the plate while preserving glyph edges
    crop_denoised = cv2.bilateralFilter(crop_gray, d=7, sigmaColor=35, sigmaSpace=35)
    
    # 3. Sauvola local thresholding (for LIGHT text on DARK background)
    ws = 31
    k = 0.12
    thresh = threshold_sauvola(crop_denoised, window_size=ws, k=k)
    crop_bin = (crop_denoised > thresh).astype(np.uint8) * 255
    
    # 4. Character-level segmentation using connected components
    dilated_crop = cv2.dilate(crop_bin, np.ones((2, 2), np.uint8))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated_crop, connectivity=8)
    
    # Bounding box constraints for individual character glyphs
    min_area = 12
    max_area = int((cx1 - cx0) * (cy1 - cy0) * 0.02)
    
    crop_canvas = np.zeros_like(crop_bin)
    pad = 2
    valid_glyphs = 0
    
    # Extract glyphs one by one
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
            
        gx = int(stats[label, cv2.CC_STAT_LEFT])
        gy = int(stats[label, cv2.CC_STAT_TOP])
        gw = int(stats[label, cv2.CC_STAT_WIDTH])
        gh = int(stats[label, cv2.CC_STAT_HEIGHT])
        
        # Filter border artifacts (remove anything touching the crop edges)
        if gx <= 2 or gy <= 2 or (gx + gw) >= (cx1 - cx0 - 2) or (gy + gh) >= (cy1 - cy0 - 2):
            continue
            
        # Stamp characters onto clean crop canvas
        gx0 = max(0, gx - pad)
        gy0 = max(0, gy - pad)
        gx1 = min(cx1 - cx0, gx + gw + pad)
        gy1 = min(cy1 - cy0, gy + gh + pad)
        
        glyph_crop = crop_bin[gy0:gy1, gx0:gx1]
        crop_canvas[gy0:gy1, gx0:gx1] = np.maximum(crop_canvas[gy0:gy1, gx0:gx1], glyph_crop)
        valid_glyphs += 1
        
    print(f"Extracted and stamped {valid_glyphs} character glyphs.")
    
    # 5. Build full output canvas (matching original dimensions)
    canvas = np.zeros((H, W), dtype=np.uint8)
    canvas[cy0:cy1, cx0:cx1] = crop_canvas
    
    # Final morphological close and cleanup
    canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    
    out_path = Path("tune_img334_out/image9_segmented.png")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(out_path), canvas)
    print(f"Binarisation output saved to: {out_path}")

if __name__ == "__main__":
    main()
