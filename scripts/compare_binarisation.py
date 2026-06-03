import logging
from pathlib import Path
import cv2
import numpy as np
from skimage.filters import threshold_sauvola

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
LOGGER = logging.getLogger("compare_binarise")

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/comparison_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Select test images
TEST_IMAGES = [
    {"lang": "kannada", "path": RAW_DIR / "kannada_stone" / "image2.jpeg"},
    {"lang": "malayalam", "path": RAW_DIR / "malayalam_stone" / "image1.jpeg"},
    {"lang": "tamil", "path": RAW_DIR / "tamil_stone" / "tamil_001.jpg"},
    {"lang": "telugu", "path": RAW_DIR / "telugu_stone" / "image2.jpg"},
    {"lang": "tulu", "path": RAW_DIR / "tulu_stone" / "image5.png"},
]

# ─── Original Implementations ────────────────────────────────────────────────

def original_binarise_stone(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape
    smooth = cv2.GaussianBlur(gray, (0, 0), sigmaX=5, sigmaY=5)
    k = max(31, min(h, w) // 12)
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    black_hat = cv2.morphologyEx(smooth, cv2.MORPH_BLACKHAT, kernel)
    black_hat = cv2.normalize(black_hat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    thresh_val = int(np.percentile(black_hat, 75))
    thresh_val = max(thresh_val, 30)
    _, binary = cv2.threshold(black_hat, thresh_val, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return binary

def original_binarise_palm_leaf(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 5
    )
    binary = cv2.bitwise_not(binary)
    h, w = binary.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    for corner in [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]:
        cv2.floodFill(binary, mask, (corner[1], corner[0]), 0)
    kernel = np.ones((2, 2), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary

# ─── Improved/Adaptive Implementations ────────────────────────────────────────

def adaptive_binarise_stone(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape
    
    # 1. Dynamically scale sigma based on image size
    sigma = max(0.5, min(h, w) / 600.0)
    smooth = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    
    # 2. Dynamically scale black-hat kernel size k
    k = max(5, min(h, w) // 15)
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    black_hat = cv2.morphologyEx(smooth, cv2.MORPH_BLACKHAT, kernel)
    black_hat = cv2.normalize(black_hat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 3. Sauvola local thresholding on black-hat image instead of global percentile!
    # This detects strokes locally and avoids hardcoding the foreground percentage.
    ws = max(15, min(h, w) // 12)
    if ws % 2 == 0:
        ws += 1
    thresh = threshold_sauvola(black_hat, window_size=ws, k=0.2)
    binary = (black_hat > thresh).astype(np.uint8) * 255
    
    # 4. Adaptive morphological cleaning
    op_sz = max(1, min(h, w) // 300)
    cl_sz = max(3, min(h, w) // 150)
    if op_sz % 2 == 0: op_sz += 1
    if cl_sz % 2 == 0: cl_sz += 1
    
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((op_sz, op_sz), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((cl_sz, cl_sz), np.uint8))
    return binary

def true_binarise_palm_leaf(img: np.ndarray) -> np.ndarray:
    # 1. Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    
    # 2. Apply CLAHE to L channel to normalise illumination
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_norm = clahe.apply(L)
    
    # 3. Apply Sauvola local thresholding on L channel (ink is dark)
    h, w = L_norm.shape
    ws = max(15, min(h, w) // 10)
    if ws % 2 == 0:
        ws += 1
    thresh_L = threshold_sauvola(L_norm, window_size=ws, k=0.15)
    binary_L = (L_norm < thresh_L).astype(np.uint8) * 255
    
    # 4. Create an A-channel Otsu mask to suppress warm background fibre texture
    A_blur = cv2.GaussianBlur(A, (5, 5), 0)
    _, mask_A = cv2.threshold(A_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 5. Intersect Sauvola result with the A-channel mask
    binary = cv2.bitwise_and(binary_L, mask_A)
    
    # 6. Morphological cleanup
    kernel = np.ones((2, 2), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary

# ─── Main Comparison ────────────────────────────────────────────────────────

def main():
    LOGGER.info("Starting binarisation comparison...")
    for item in TEST_IMAGES:
        lang = item["lang"]
        img_path = item["path"]
        
        if not img_path.exists():
            continue
            
        LOGGER.info("---------------------------------------------")
        LOGGER.info("Processing %s: %s", lang, img_path.name)
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        img_out_dir = OUT_DIR / lang
        img_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Test original stone
        bin_orig_stone = original_binarise_stone(img)
        cv2.imwrite(str(img_out_dir / f"{img_path.stem}_orig_stone.png"), bin_orig_stone)
        LOGGER.info("Original Stone FG%%: %.1f%%", (cv2.countNonZero(bin_orig_stone)/bin_orig_stone.size)*100)
        
        # Test adaptive stone
        bin_adap_stone = adaptive_binarise_stone(img)
        cv2.imwrite(str(img_out_dir / f"{img_path.stem}_adap_stone.png"), bin_adap_stone)
        LOGGER.info("Adaptive Stone FG%%: %.1f%%", (cv2.countNonZero(bin_adap_stone)/bin_adap_stone.size)*100)
        
        # Test original palm leaf
        bin_orig_palm = original_binarise_palm_leaf(img)
        cv2.imwrite(str(img_out_dir / f"{img_path.stem}_orig_palm.png"), bin_orig_palm)
        LOGGER.info("Original Palm Leaf FG%%: %.1f%%", (cv2.countNonZero(bin_orig_palm)/bin_orig_palm.size)*100)
        
        # Test true palm leaf
        bin_true_palm = true_binarise_palm_leaf(img)
        cv2.imwrite(str(img_out_dir / f"{img_path.stem}_true_palm.png"), bin_true_palm)
        LOGGER.info("True Palm Leaf FG%%: %.1f%%", (cv2.countNonZero(bin_true_palm)/bin_true_palm.size)*100)

    LOGGER.info("Comparison complete. Outputs saved in data/comparison_outputs/")

if __name__ == "__main__":
    main()
