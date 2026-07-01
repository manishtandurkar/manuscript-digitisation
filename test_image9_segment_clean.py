import cv2
import numpy as np
from pathlib import Path
from skimage.filters import threshold_sauvola

def main():
    img_path = r"C:\6th semester EL's\Interdisciplinary project\Implementation\manuscript-digitisation\data\raw\malayalam_stone\image9.png"
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("Failed to load image9.png")
        return

    H, W = img.shape[:2]
    shorter = min(H, W)
    
    # 1. Convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Extract the rectangular copper plate mask
    # The plate is dark brown (gray < 180), background is bright white (gray > 200)
    _, plate_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    
    # Clean up the plate mask using morphological close then open
    k_size = max(5, shorter // 30)
    plate_mask = cv2.morphologyEx(plate_mask, cv2.MORPH_CLOSE, np.ones((k_size, k_size), np.uint8))
    plate_mask = cv2.morphologyEx(plate_mask, cv2.MORPH_OPEN, np.ones((k_size, k_size), np.uint8))
    
    # Find bounding box of the plate
    contours, _ = cv2.findContours(plate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Could not find copper plate contour")
        return
        
    # Get the largest contour (the plate)
    large_contour = max(contours, key=cv2.contourArea)
    px, py, pw, ph = cv2.boundingRect(large_contour)
    print(f"Copper plate bounding box: X={px}, Y={py}, W={pw}, H={ph}")
    
    # 3. Crop to the plate bounding box
    # Let's crop slightly inward (e.g. 5 pixels) to completely exclude the outer white border
    border_inset = 6
    cx0 = max(0, px + border_inset)
    cy0 = max(0, py + border_inset)
    cx1 = min(W, px + pw - border_inset)
    cy1 = min(H, py + ph - border_inset)
    
    crop_gray = gray[cy0:cy1, cx0:cx1]
    
    # Bilateral filter on the cropped region to smooth texture but preserve glyph edges
    crop_denoised = cv2.bilateralFilter(crop_gray, d=7, sigmaColor=35, sigmaSpace=35)
    
    # 4. Sauvola local thresholding (for light text on dark background)
    ws = 31
    k = 0.12
    thresh = threshold_sauvola(crop_denoised, window_size=ws, k=k)
    crop_bin = (crop_denoised > thresh).astype(np.uint8) * 255
    
    # 5. Character-level segmentation (Connected Components) on the cropped region
    # Dilate slightly to connect character strokes
    dilated_crop = cv2.dilate(crop_bin, np.ones((2, 2), np.uint8))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated_crop, connectivity=8)
    
    # Bounding box limits for a character glyph
    min_area = 12
    max_area = int((cx1 - cx0) * (cy1 - cy0) * 0.02) # individual characters should be small
    
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
        
        # Avoid edge artifacts: if a component touches the boundary of the cropped plate,
        # it is likely part of the plate border or noise, so we skip it.
        if gx <= 2 or gy <= 2 or (gx + gw) >= (cx1 - cx0 - 2) or (gy + gh) >= (cy1 - cy0 - 2):
            continue
            
        # Crop glyph from binarised image
        gx0 = max(0, gx - pad)
        gy0 = max(0, gy - pad)
        gx1 = min(cx1 - cx0, gx + gw + pad)
        gy1 = min(cy1 - cy0, gy + gh + pad)
        
        glyph_crop = crop_bin[gy0:gy1, gx0:gx1]
        
        # Stamp onto cropped canvas
        crop_canvas[gy0:gy1, gx0:gx1] = np.maximum(crop_canvas[gy0:gy1, gx0:gx1], glyph_crop)
        valid_glyphs += 1
        
    print(f"Successfully extracted and stamped {valid_glyphs} characters.")
    
    # 6. Reconstruct the full output canvas (matching original image size)
    canvas = np.zeros((H, W), dtype=np.uint8)
    canvas[cy0:cy1, cx0:cx1] = crop_canvas
    
    # 7. Final Morphological close and minor cleanup
    canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    
    out_path = Path("tune_img334_out/image9_segmented_clean.png")
    out_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    print(f"Saved cleaned segmented characters to {out_path}")

if __name__ == "__main__":
    main()
