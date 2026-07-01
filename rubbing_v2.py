import cv2
import numpy as np
from pathlib import Path

def binarise_rubbing_v2(gray: np.ndarray) -> np.ndarray:
    """
    Dynamically-scaling Close-Open Morphological Binarisation pipeline
    optimized for rubbing/estampage images of any resolution (large or small).
    """
    H_orig, W_orig = gray.shape
    shorter = min(H_orig, W_orig)
    is_low_res = shorter < 300
    
    # 1. Bilateral Filter to smooth stone texture but preserve edges
    # For low-res images, texture is coarse relative to character size, so filter gently
    d = 5 if is_low_res else 9
    sigma_color = 50 if is_low_res else 75
    sigma_space = 50 if is_low_res else 75
    smoothed = cv2.bilateralFilter(gray, d, sigma_color, sigma_space)
    
    # 2. Upscale low-res image by 4x using Lanczos interpolation
    if is_low_res:
        scale = 4
        smoothed = cv2.resize(smoothed, (W_orig * scale, H_orig * scale), interpolation=cv2.INTER_LANCZOS4)
        # Antialias/smooth out interpolation artifacts before thresholding
        smoothed = cv2.GaussianBlur(smoothed, (3, 3), 0)
    else:
        scale = 1
        
    H, W = smoothed.shape
    
    # 3. Enhance contrast
    # Use CLAHE to make text stand out against local background variations
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(smoothed)
    
    # 4. Adaptive/Local Background Subtraction
    # Compute a local background using a large Gaussian blur and subtract it
    bg_size = 51 if is_low_res else 101
    if bg_size % 2 == 0: bg_size += 1
    bg = cv2.GaussianBlur(enhanced, (bg_size, bg_size), 0)
    subtracted = cv2.subtract(enhanced, bg)
    subtracted = cv2.normalize(subtracted, None, 0, 255, cv2.NORM_MINMAX)
    
    # 5. Thresholding - use Otsu's on the normalized subtraction
    _, binary = cv2.threshold(subtracted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 6. Connected Component Analysis (CCA) for character-by-character filtering
    # We run this on the raw thresholded binary to prevent early merging of adjacent characters
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    canvas = np.zeros_like(binary)
    
    # Adjust thresholds based on scale (raw characters are smaller before closing)
    min_area = 35 if is_low_res else 120
    max_area = 3000 if is_low_res else 8000
    min_dim = 5 if is_low_res else 10
    max_aspect = 5.0
    
    valid_count = 0
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        gx = stats[label, cv2.CC_STAT_LEFT]
        gy = stats[label, cv2.CC_STAT_TOP]
        gw = stats[label, cv2.CC_STAT_WIDTH]
        gh = stats[label, cv2.CC_STAT_HEIGHT]
        
        # Filter size
        if area < min_area or area > max_area or gw < min_dim or gh < min_dim:
            continue
            
        # Filter aspect ratio (reject long thin lines like cracks)
        aspect = max(gw / gh, gh / gw) if gh > 0 and gw > 0 else 1.0
        if aspect > max_aspect:
            continue
            
        # Filter border components
        edge_dist = 2 if is_low_res else 5
        if gx <= edge_dist or gy <= edge_dist or (gx + gw) >= (W - edge_dist) or (gy + gh) >= (H - edge_dist):
            continue
            
        # Extent filter (area / bounding box area) - character components should not be extremely sparse or solid blocks
        extent = area / (gw * gh)
        if extent < 0.22 or extent > 0.85:
            continue
            
        canvas[labels == label] = 255
        valid_count += 1
        
    print(f"Binarised (is_low_res={is_low_res}): extracted {valid_count} characters.")
    
    # 7. Post-processing to make characters look smooth, solid, and connected
    if is_low_res:
        # Close internal gaps in character strokes then gently dilate to clean and thicken
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        canvas = cv2.dilate(canvas, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
    else:
        # High resolution doesn't need heavy morph, just minor dilation if at all
        canvas = cv2.dilate(canvas, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        
    return canvas

def main():
    img_path = Path(
        r"C:\6th semester EL's\Interdisciplinary project\Implementation\manuscript-digitisation"
        r"\data\binarised_representative_samples\kannada_stone\image3_original.jpeg"
    )

    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"FAILED TO LOAD IMAGE: {img_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = binarise_rubbing_v2(gray)

    out_path = img_path.parent / "image3_rubbing_v2.png"
    cv2.imwrite(str(out_path), result)
    print(f"Done. Output saved to: {out_path}")

if __name__ == "__main__":
    main()