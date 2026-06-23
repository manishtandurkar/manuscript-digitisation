import cv2
import numpy as np
from pathlib import Path
from skimage.filters import threshold_sauvola

def remove_noise_blobs(binary: np.ndarray, min_size: int = 10, min_length: int = 5) -> np.ndarray:
    """Remove small disconnected noise components from the binary image."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area >= min_size or max(cw, ch) >= min_length:
            cleaned[labels == label] = 255
    return cleaned

def binarise_malayalam_image1(img: np.ndarray) -> np.ndarray:
    """
    Refined binarisation logic for malayalam_stone/image1_original.jpeg.
    Fixes the issue of missing characters by tuning component thresholds,
    border margins, and morphological operations.
    """
    H, W = img.shape[:2]
    shorter = min(H, W)

    # 1. Extract Green channel (gives great contrast for the palm leaf ink)
    g = img[:, :, 1]
    
    # 2. Smooth rock/leaf texture using bilateral filter while keeping character edges crisp
    sigma_s = max(5, shorter // 30)
    denoised = cv2.bilateralFilter(g, d=9, sigmaColor=30, sigmaSpace=sigma_s)
    
    # 3. Enhance local contrast with CLAHE
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)
    
    # 4. Sauvola local thresholding to handle lighting variations
    ws = max(21, min(61, (shorter // 8) | 1))
    if ws % 2 == 0:
        ws += 1
    thresh = threshold_sauvola(enhanced, window_size=ws, k=0.18)
    binary = (enhanced < thresh).astype(np.uint8) * 255
    
    # 5. Flood fill from the corners of the raw binary to clear outer border noise
    # We do this before dilation to avoid merging text characters with the border frame.
    mask = np.zeros((H + 2, W + 2), np.uint8)
    for corner in [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]:
        if binary[corner[0], corner[1]] == 255:
            cv2.floodFill(binary, mask, (corner[1], corner[0]), 0)
        
    # 6. Connected Component Analysis (CCA) to filter remaining non-character components
    # Dilate slightly to group character fragments together for stable bounding boxes
    dil_k = max(3, shorter // 40)
    dilated = cv2.dilate(binary, np.ones((dil_k, dil_k), np.uint8))
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    
    min_area = 10   # Keep small components/vowel markers
    max_area = int(H * W * 0.40)
    pad = 2
    canvas = np.zeros((H, W), dtype=np.uint8)
    
    # We use a smaller edge distance (e.g., 4 pixels) to avoid clipping text components
    # near the left/right/top/bottom margins.
    edge_dist = 4
    
    kept = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue

        x  = int(stats[label, cv2.CC_STAT_LEFT])
        y  = int(stats[label, cv2.CC_STAT_TOP])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        
        # Filter components that are right on the edge of the leaf/image border
        if x <= edge_dist or y <= edge_dist or (x + cw) >= (W - edge_dist) or (y + ch) >= (H - edge_dist):
            continue
            
        x0 = max(0, x - pad);  y0 = max(0, y - pad)
        x1 = min(W, x + cw + pad); y1 = min(H, y + ch + pad)

        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            continue

        # Local binarisation inside each component's bounding box
        g_crop = crop[:, :, 1]
        cr_h, cr_w = g_crop.shape
        if cr_h < 6 or cr_w < 6:
            _, local_bin = cv2.threshold(g_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            ws_c = max(5, min(25, (min(cr_h, cr_w) // 2) | 1))
            if ws_c % 2 == 0:
                ws_c += 1
            thresh_c = threshold_sauvola(g_crop, window_size=ws_c, k=0.15)
            local_bin = (g_crop < thresh_c).astype(np.uint8) * 255

        # Stamp the high-quality local binarisation back onto the canvas
        canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], local_bin)
        kept += 1

    print(f"Extracted {kept} text components.")

    # 7. Post-processing to close minor gaps and remove tiny noise specks
    canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    canvas = remove_noise_blobs(canvas, min_size=8, min_length=4)
    
    return canvas

def main():
    img_path = Path(
        r"C:\6th semester EL's\Interdisciplinary project\Implementation\manuscript-digitisation"
        r"\data\binarised_representative_samples\malayalam_stone\image1_original.jpeg"
    )
    
    if not img_path.exists():
        print(f"Error: Image not found at {img_path}")
        return
        
    img = cv2.imread(str(img_path))
    if img is None:
        print("Error: Could not read image.")
        return
        
    print(f"Loaded image of shape: {img.shape}")
    result = binarise_malayalam_image1(img)
    
    out_path = img_path.parent / "image1_rubbing_v3.png"
    cv2.imwrite(str(out_path), result)
    print(f"Successfully saved output to: {out_path}")

if __name__ == "__main__":
    main()
