import cv2
import numpy as np
from pathlib import Path

def binarise_image1_dedicated(img: np.ndarray) -> np.ndarray:
    """
    Dedicated refined binarisation for raw/kannada_stone/image1.jpeg.
    Outputs: white text, black background.
    """
    b, g, r = cv2.split(img)
    H_orig, W_orig = g.shape
    scale = 4
    
    # 1. Upscale first to 4x using Lanczos
    upscaled = cv2.resize(g, (W_orig * scale, H_orig * scale), interpolation=cv2.INTER_LANCZOS4)
    
    # 2. Bilateral filter on upscaled image to smooth rock grain
    smoothed = cv2.bilateralFilter(upscaled, d=5, sigmaColor=50, sigmaSpace=50)
    smoothed = cv2.GaussianBlur(smoothed, (3, 3), 0)
    
    # 3. CLAHE local contrast normalisation
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(smoothed)
    
    # 4. Local Subtract (Valley detection: bg - enhanced)
    bg = cv2.GaussianBlur(enhanced, (51, 51), 0)
    subtracted = cv2.subtract(bg, enhanced)
    subtracted = cv2.normalize(subtracted, None, 0, 255, cv2.NORM_MINMAX)
    
    # 5. Otsu thresholding
    _, binary = cv2.threshold(subtracted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 6. Connected Component Analysis (CCA)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    canvas = np.zeros_like(binary)
    
    min_area = 30
    max_area = 2500
    min_dim = 5
    edge_dist = 2
    
    kept_count = 0
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        gx = stats[label, cv2.CC_STAT_LEFT]
        gy = stats[label, cv2.CC_STAT_TOP]
        gw = stats[label, cv2.CC_STAT_WIDTH]
        gh = stats[label, cv2.CC_STAT_HEIGHT]
        
        # Filter size
        if area < min_area or area > max_area or gw < min_dim or gh < min_dim:
            continue
            
        # Aspect ratio checks (strata / crack filters)
        aspect_w_h = gw / gh
        aspect_h_w = gh / gw
        if aspect_w_h > 2.0 or aspect_h_w > 2.5:
            continue
            
        # Border check
        if gx <= edge_dist or gy <= edge_dist or (gx + gw) >= (binary.shape[1] - edge_dist) or (gy + gh) >= (binary.shape[0] - edge_dist):
            continue
            
        # Extent filter
        extent = area / (gw * gh)
        if extent < 0.20 or extent > 0.85:
            continue
            
        # ROI filter: only keep components in the text band to remove background rock seam noise
        # Middle text band: Y in [130, 600], X in [80, 1050]
        if gy < 130 or gy + gh > 600 or gx < 80 or gx + gw > 1050:
            continue
            
        canvas[labels == label] = 255
        kept_count += 1
        
    print(f"Binarised image1: extracted {kept_count} characters.")
    
    # 7. Post-processing to connect and smooth characters
    canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    canvas = cv2.dilate(canvas, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
    
    return canvas

def main():
    img_path = Path(
        r"C:\6th semester EL's\Interdisciplinary project\Implementation\manuscript-digitisation"
        r"\data\raw\kannada_stone\image1.jpeg"
    )
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"FAILED TO LOAD IMAGE: {img_path}")
        return
        
    result = binarise_image1_dedicated(img)
    
    # Save output in raw directory next to original for easy review
    out_path = img_path.parent / "image1_binarised_refined.png"
    cv2.imwrite(str(out_path), result)
    print(f"Done. Output saved to: {out_path}")

if __name__ == "__main__":
    main()
