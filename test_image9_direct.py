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
    
    # 2. Bilateral filter to smooth texture while preserving character boundaries
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=50, sigmaSpace=50)
    
    # 3. Apply Sauvola (looking for light text on dark plate)
    ws = 51
    k = 0.15
    thresh = threshold_sauvola(denoised, window_size=ws, k=k)
    binary = (denoised > thresh).astype(np.uint8) * 255
    
    # 4. Remove outer white border using flood fill from corners
    # Create flood fill mask
    flood_mask = np.zeros((H + 2, W + 2), np.uint8)
    
    # Since the outer background is white, we flood fill it with black from the corners
    # Let's check the corners first to see if they are white (near 255)
    # We flood fill from each corner (0,0), (0, W-1), (H-1, 0), (H-1, W-1)
    filled = binary.copy()
    for corner in [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]:
        # Flood fill white region at the corner to 0 (black)
        # We check if pixel at corner is white in the binary image
        if filled[corner[0], corner[1]] == 255:
            cv2.floodFill(filled, flood_mask, (corner[1], corner[0]), 0)

    # 5. Clean up small noise blobs
    # Since characters are relatively small on a 253x704 image, we can use a smaller min_size
    from src.binarise import remove_noise_blobs
    cleaned = remove_noise_blobs(filled, min_size=15, min_length=8)
    
    # 6. Morphological close to heal character strokes
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

    cv2.imwrite("tune_img334_out/image9_direct.png", cleaned)
    print("Saved tune_img334_out/image9_direct.png")

if __name__ == "__main__":
    main()
