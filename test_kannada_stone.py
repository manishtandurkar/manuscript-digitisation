import cv2
import numpy as np
from pathlib import Path

def binarise_stone_morphology_clean(img, gray):
    """
    Highly refined Morphological Close-Open Binarisation pipeline
    optimized for black-and-white stone rubbings/estampages.
    
    Steps:
      1. Grayscale closing to fill the hollow centers of the letters.
      2. Background subtraction to level uneven lighting gradients.
      3. Grayscale opening to completely erase background speckle noise.
      4. Otsu's thresholding.
      5. Connected component filtering to remove remaining small dots and line cracks.
    """
    H, W = gray.shape
    print("\n--- Running Close-Open Morphological Binarisation ---")
    
    # 1. Grayscale closing with a 7x7 circular kernel to merge hollow outlines into solid shapes
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, close_kernel)
    
    # 2. Subtract local background to flatten global lighting/contrast variations
    bg_kernel_size = 101
    bg = cv2.GaussianBlur(closed, (bg_kernel_size, bg_kernel_size), 0)
    subtracted = cv2.subtract(closed, bg)
    subtracted = cv2.normalize(subtracted, None, 0, 255, cv2.NORM_MINMAX)
    
    # 3. Grayscale opening with a 5x5 circular kernel to pre-erase small noise speckles
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(subtracted, cv2.MORPH_OPEN, open_kernel)
    
    # 4. Otsu's thresholding to get the binary mask (since gradients and noise are gone)
    _, binary = cv2.threshold(opened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 5. Connected Component Analysis to remove remaining noise and cracks
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    canvas = np.zeros_like(binary)
    
    # Strict thresholds for valid character glyphs at this resolution
    min_area = 95       # Erase any stamped blobs smaller than 95px (removes background speckles)
    min_dim = 10        # Both width and height must be >= 10px (valid characters)
    max_area = 5000     # Exclude giant artifacts
    max_aspect_ratio = 2.5 # Exclude long vertical/horizontal crack lines
    
    valid_count = 0
    removed_by_area = 0
    removed_by_aspect = 0
    removed_by_border = 0
    
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        gx = int(stats[label, cv2.CC_STAT_LEFT])
        gy = int(stats[label, cv2.CC_STAT_TOP])
        gw = int(stats[label, cv2.CC_STAT_WIDTH])
        gh = int(stats[label, cv2.CC_STAT_HEIGHT])
        
        # A. Filter by area and minimum dimensions
        if area < min_area or area > max_area or gw < min_dim or gh < min_dim:
            removed_by_area += 1
            continue
            
        # B. Filter by aspect ratio (rejects crack lines)
        aspect_ratio = max(gw / gh, gh / gw) if gh > 0 and gw > 0 else 1.0
        if aspect_ratio > max_aspect_ratio:
            removed_by_aspect += 1
            continue
            
        # C. Filter border noise (reject anything touching the very edges of the image)
        if gx <= 5 or gy <= 5 or (gx + gw) >= (W - 5) or (gy + gh) >= (H - 5):
            removed_by_border += 1
            continue
            
        # Stamp valid component
        canvas[labels == label] = 255
        valid_count += 1
        
    print(f"Components Cleanup Stats:")
    print(f"  - Total components found: {num_labels - 1}")
    print(f"  - Valid characters stamped: {valid_count}")
    print(f"  - Removed by area/size limits: {removed_by_area}")
    print(f"  - Removed by aspect ratio (cracks): {removed_by_aspect}")
    print(f"  - Removed by border check (edges): {removed_by_border}")
    
    return canvas

def main():
    img_path = r"C:\6th semester EL's\Interdisciplinary project\Implementation\manuscript-digitisation\data\binarised_representative_samples\kannada_stone\image2_original.jpeg"
    output_dir = Path(r"C:\6th semester EL's\Interdisciplinary project\Implementation\manuscript-digitisation\data\binarised_representative_samples\kannada_stone")
    
    # Load image
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"Failed to load image from: {img_path}")
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Run binarisation
    cleaned_binary = binarise_stone_morphology_clean(img, gray)
    
    # Save output
    out_path = output_dir / "image2_binarised_refined.png"
    cv2.imwrite(str(out_path), cleaned_binary)
    print(f"\nSuccess! Saved refined output to: {out_path}")
    print("Please execute test_kannada_stone.py and verify the results.")

if __name__ == "__main__":
    main()
