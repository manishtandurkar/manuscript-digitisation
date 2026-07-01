import cv2
import numpy as np
from pathlib import Path

def binarise_tamil_010(img: np.ndarray) -> np.ndarray:
    """
    Dedicated character segmentation and reconstruction for tamil_010_original.jpg.
    1. HSV-based green foliage mask suppression.
    2. Morphological stone slab mask extraction and erosion.
    3. Direct Sauvola local thresholding (to detect dark text on light background).
    4. Text-specific ROI crop to remove border noise.
    5. Connected Component Analysis (CCA) size and aspect ratio filtering.
    6. Morphological closing to smooth character strokes.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 1. Segment green foliage
    lower_green = np.array([25, 30, 30])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    gray_no_green = gray.copy()
    gray_no_green[green_mask > 0] = 0

    # 2. Extract stone mask
    blurred = cv2.GaussianBlur(gray_no_green, (25, 25), 0)
    _, bright_mask = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    closed = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel_close)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    largest_label = 0
    largest_area = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest_label = i

    stone_mask = np.zeros_like(gray)
    if largest_label > 0:
        stone_mask[labels == largest_label] = 255

    stone_mask_eroded = cv2.erode(stone_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (85, 85)))

    # 3. Direct Sauvola (no CLAHE to avoid grain enhancement)
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=35, sigmaSpace=35)

    from skimage.filters import threshold_sauvola
    ws = 51
    k = 0.18
    thresh = threshold_sauvola(denoised, window_size=ws, k=k)
    raw_bin = (denoised < thresh).astype(np.uint8) * 255
    binary_masked = cv2.bitwise_and(raw_bin, stone_mask_eroded)

    # 4. Restrict to ROI (bounding box of characters on this specific slab)
    roi_mask = np.zeros_like(binary_masked)
    roi_mask[180:920, 180:880] = 255
    binary_masked = cv2.bitwise_and(binary_masked, roi_mask)

    # 5. CCA filtering
    n, labels_bin, stats_bin, _ = cv2.connectedComponentsWithStats(binary_masked, connectivity=8)
    canvas = np.zeros_like(binary_masked)
    min_area = 150
    max_area = 12000
    min_dim = 12
    max_dim = 250

    for i in range(1, n):
        area = stats_bin[i, cv2.CC_STAT_AREA]
        w = stats_bin[i, cv2.CC_STAT_WIDTH]
        h = stats_bin[i, cv2.CC_STAT_HEIGHT]
        
        if area < min_area or area > max_area:
            continue
        if w < min_dim or h < min_dim or w > max_dim or h > max_dim:
            continue
        
        aspect = max(w / h, h / w)
        if aspect > 4.5:
            continue
            
        extent = area / (w * h)
        if extent < 0.15 or extent > 0.85:
            continue

        canvas[labels_bin == i] = 255

    # 6. Apply a final morphological close to make character strokes smoother
    canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return canvas

if __name__ == "__main__":
    input_path = r"C:\6th semester EL's\Interdisciplinary project\Implementation\manuscript-digitisation\data\binarised_representative_samples\tamil_stone\tamil_010_original.jpg"
    output_path = r"C:\6th semester EL's\Interdisciplinary project\Implementation\manuscript-digitisation\data\binarised_representative_samples\tamil_stone\tamil_010_binarised_FIXED.png"
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: could not read {input_path}")
    else:
        binary = binarise_tamil_010(img)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, binary)
        print(f"Binarised output successfully saved to {output_path}")
