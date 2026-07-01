import cv2
import numpy as np
import sys
from pathlib import Path
from skimage.filters import threshold_sauvola

def main():
    img_path = r"C:\6th semester EL's\Interdisciplinary project\Implementation\manuscript-digitisation\data\raw\malayalam_stone\image9.png"
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Run the exact code in binarise_stone:
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=50, sigmaSpace=50)
    shorter = min(gray.shape)
    ws = max(51, min(151, (shorter // 10) | 1))
    if ws % 2 == 0:
        ws += 1
    k = 0.15
    thresh = threshold_sauvola(denoised, window_size=ws, k=k)
    binary = (denoised < thresh).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    
    from src.binarise import remove_noise_blobs
    binary = remove_noise_blobs(binary, min_size=20, min_length=10)
    
    print("Before polarity flip - mean:", binary.mean())
    print("Before polarity flip - corners:", [binary[0,0], binary[0,-1], binary[-1,0], binary[-1,-1]])
    
    if binary.mean() >= 127:
        binary = cv2.bitwise_not(binary)
        
    print("After polarity flip - mean:", binary.mean())
    print("After polarity flip - corners:", [binary[0,0], binary[0,-1], binary[-1,0], binary[-1,-1]])
    
    # Let's see what corners are in binary
    h_b, w_b = binary.shape[:2]
    flood_mask = np.zeros((h_b + 2, w_b + 2), np.uint8)
    for corner in [(0, 0), (0, w_b - 1), (h_b - 1, 0), (h_b - 1, w_b - 1)]:
        val = binary[corner[0], corner[1]]
        print(f"Checking corner {corner}: value={val}")
        if val == 255:
            # Run floodfill
            cv2.floodFill(binary, flood_mask, (corner[1], corner[0]), 0)
            
    print("After floodfill - corners:", [binary[0,0], binary[0,-1], binary[-1,0], binary[-1,-1]])
    print("After floodfill - mean:", binary.mean())

if __name__ == "__main__":
    main()
