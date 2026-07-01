import cv2
import numpy as np
from pathlib import Path

def binarise_tamil_026_segmental(img: np.ndarray, method: str = "segmental") -> np.ndarray:
    """
    Binarisation logic for tamil_026_original.jpg.
    
    Supports:
    1. 'segmental': Uses local Sauvola parameters (ws=15, k=0.25) to isolate fine character details,
                    morphological closing, CCA size filtering (12 to 2000 px) to remove noise,
                    and border floodfilling to clean scanner edges.
    2. 'stone_default': Uses bilateral filtering (d=5), Sauvola (ws=25, k=0.12), morphological closing,
                        and noise cleaning, matching the main binarise_stone script.
                        
    Output: black characters on white background.
    """
    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    H, W = gray.shape[:2]

    if method == "segmental":
        # Sauvola thresholding with ws=15, k=0.25 (keeps components small & isolated)
        from skimage.filters import threshold_sauvola
        thresh = threshold_sauvola(gray, window_size=15, k=0.25)
        binary = (gray < thresh).astype(np.uint8) * 255

        # Morph close (2x2)
        binary_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

        # Connected Components Analysis filtering
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_close, connectivity=8)
        canvas = np.zeros_like(binary_close)
        min_area = 12
        max_area = 2000

        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if min_area <= area <= max_area:
                canvas[labels == label] = 255

        # Flood fill from borders to wipe out border scanner noise
        flood_mask = np.zeros((H + 2, W + 2), np.uint8)
        for x in range(W):
            for y in [0, H - 1]:
                if canvas[y, x] == 255:
                    cv2.floodFill(canvas, flood_mask, (x, y), 0)
        for y in range(H):
            for x in [0, W - 1]:
                if canvas[y, x] == 255:
                    cv2.floodFill(canvas, flood_mask, (x, y), 0)

        # Invert to produce black characters on white background
        return cv2.bitwise_not(canvas)
        
    else:  # "stone_default"
        # 1. Bilateral filter
        denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=30, sigmaSpace=30)
        
        # 2. Sauvola ws=25, k=0.12
        from skimage.filters import threshold_sauvola
        thresh = threshold_sauvola(denoised, window_size=25, k=0.12)
        binary = (denoised < thresh).astype(np.uint8) * 255
        
        # 3. Morph close (2x2)
        binary_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        
        # 4. Remove noise blobs (min_size=12, min_length=6)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_close, connectivity=8)
        canvas = np.zeros_like(binary_close)
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            cw = int(stats[label, cv2.CC_STAT_WIDTH])
            ch = int(stats[label, cv2.CC_STAT_HEIGHT])
            if area >= 12 or max(cw, ch) >= 6:
                canvas[labels == label] = 255
                
        # 5. Flood fill borders
        flood_mask = np.zeros((H + 2, W + 2), np.uint8)
        for x in range(W):
            for y in [0, H - 1]:
                if canvas[y, x] == 255:
                    cv2.floodFill(canvas, flood_mask, (x, y), 0)
        for y in range(H):
            for x in [0, W - 1]:
                if canvas[y, x] == 255:
                    cv2.floodFill(canvas, flood_mask, (x, y), 0)
                    
        # Invert to produce black characters on white background
        return cv2.bitwise_not(canvas)

if __name__ == "__main__":
    input_path = r"data/binarised_representative_samples/tamil_stone/tamil_026_original.jpg"
    output_path = r"data/binarised_representative_samples/tamil_stone/tamil_026_binarised_FIXED.png"
    
    src_candidates = [
        Path(input_path),
        Path(__file__).parent / "data" / "binarised_representative_samples" / "tamil_stone" / "tamil_026_original.jpg",
        Path(__file__).parent / "data" / "raw" / "tamil_stone" / "tamil_026.jpg",
        Path(__file__).parent / "data" / "representative_samples" / "tamil_stone" / "tamil_026.jpg"
    ]
    
    img = None
    for cand in src_candidates:
        if cand.exists():
            img = cv2.imread(str(cand))
            if img is not None:
                print(f"Loaded image from {cand}")
                break
                
    if img is None:
        print("Error: Could not load tamil_026 image.")
    else:
        # We run the 'segmental' method by default (user's requested approach)
        binary = binarise_tamil_026_segmental(img, method="segmental")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, binary)
        print(f"Saved binarised output to: {output_path}")
        print("Method 'segmental' completed successfully.")
