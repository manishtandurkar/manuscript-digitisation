import cv2
import numpy as np
import sys
from pathlib import Path

def binarise_stone_improved(img: np.ndarray, window_size: int = 101, k: float = 0.15) -> np.ndarray:
    """
    Improved binarisation for textured stone inscriptions.
    Smoothes high-frequency stone texture using bilateral filter (no CLAHE),
    then applies Sauvola local thresholding.
    """
    # 1. Convert to grayscale if BGR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    
    # 2. Bilateral filter to smooth texture while preserving character boundaries
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=50, sigmaSpace=50)
    
    # 3. Apply Sauvola local thresholding (without CLAHE, preventing noise amplification)
    from skimage.filters import threshold_sauvola
    thresh = threshold_sauvola(denoised, window_size=window_size, k=k)
    binary = (denoised < thresh).astype(np.uint8) * 255
    
    # 4. Remove small noise components (salt and pepper)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    
    # Scale minimum component size based on image height
    h, w = binary.shape
    min_size = max(20, (min(h, w) // 100) ** 2) # e.g. for 1000px shorter side, min_size is 100 pixels
    min_length = max(10, min(h, w) // 80)       # min dimension of component
    
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area >= min_size or max(cw, ch) >= min_length:
            cleaned[labels == label] = 255
            
    return cleaned

def main():
    if len(sys.argv) < 3:
        input_path = r"data/raw/Vijay Kumar extra images/img334.jpg"
        output_path = r"tune_img334_out/improved_binarisation.png"
        print(f"No arguments provided. Using defaults:\n  Input: {input_path}\n  Output: {output_path}")
    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        
    print(f"Reading {input_path}...")
    img = cv2.imdecode(np.fromfile(input_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: Could not load image from {input_path}")
        return
        
    print("Processing...")
    binary = binarise_stone_improved(img)
    
    # Save output
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), binary)
    print(f"Success! Saved binary output to {out}")

if __name__ == "__main__":
    main()
