import cv2
import numpy as np
from pathlib import Path
import os

def main():
    img_path = r"C:\6th semester EL's\Interdisciplinary project\Implementation\manuscript-digitisation\data\raw\Vijay Kumar extra images\img334.jpg"
    out_dir = Path("tune_img334_out")
    out_dir.mkdir(exist_ok=True, parents=True)

    print(f"Loading image from {img_path}...")
    if not os.path.exists(img_path):
        print("ERROR: Image does not exist!")
        return

    # Read image using imdecode to handle any unicode path issues
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("ERROR: Failed to load image!")
        return

    h, w, c = img.shape
    print(f"Image loaded successfully. Dimensions: {w}x{h}, Channels: {c}")

    # Compute basic stats
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_val = gray.mean()
    std_val = gray.std()
    print(f"Grayscale Stats - Mean: {mean_val:.2f}, Std: {std_val:.2f}")

    # Save gray image as reference
    cv2.imwrite(str(out_dir / "00_gray.png"), gray)

    # Let's import skimage if available
    try:
        from skimage.filters import threshold_sauvola, frangi
        print("Scikit-image filters loaded successfully.")
    except ImportError:
        print("WARNING: scikit-image filters not available.")
        threshold_sauvola = None
        frangi = None

    # Let's save a preprocessed version (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    cv2.imwrite(str(out_dir / "01_clahe.png"), gray_eq)

    # Method 1: Standard Otsu
    _, otsu = cv2.threshold(gray_eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite(str(out_dir / "02_otsu.png"), otsu)

    # Method 2: Adaptive Gaussian
    adaptive_gauss = cv2.adaptiveThreshold(
        gray_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10
    )
    cv2.imwrite(str(out_dir / "03_adaptive_gauss.png"), adaptive_gauss)

    # Method 3: Sauvola (if available) with different parameters
    if threshold_sauvola is not None:
        for ws in [25, 51, 101]:
            for k in [0.1, 0.15, 0.2]:
                thresh = threshold_sauvola(gray_eq, window_size=ws, k=k)
                binary = (gray_eq < thresh).astype(np.uint8) * 255
                cv2.imwrite(str(out_dir / f"04_sauvola_ws{ws}_k{k}.png"), binary)

    # Method 4: Frangi filter based (stone method)
    if frangi is not None:
        # Frangi on grayscale / CLAHE
        vessel = frangi(gray_eq.astype(np.float32) / 255.0, sigmas=range(1, 6), black_ridges=True)
        vn = cv2.normalize(vessel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(str(out_dir / "05_frangi_raw.png"), vn)
        
        # Threshold frangi output
        _, frangi_bin = cv2.threshold(vn, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(str(out_dir / "05_frangi_otsu.png"), frangi_bin)

    print("All basic runs done. Check tune_img334_out/ directory.")

if __name__ == "__main__":
    main()
