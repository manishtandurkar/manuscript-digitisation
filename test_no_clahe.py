import cv2
import numpy as np
from pathlib import Path

def main():
    img_path = r"C:\6th semester EL's\Interdisciplinary project\Implementation\manuscript-digitisation\data\raw\Vijay Kumar extra images\img334.jpg"
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("Failed to load")
        return

    out_dir = Path("tune_img334_out/no_clahe")
    out_dir.mkdir(exist_ok=True, parents=True)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Bilateral filter to smooth texture while keeping character edges
    # d=9, sigmaColor=75, sigmaSpace=75 is standard, let's try a few
    bilat = cv2.bilateralFilter(gray, 9, 50, 50)
    cv2.imwrite(str(out_dir / "bilat_9_50_50.png"), bilat)
    
    # 2. Gaussian blur
    gauss = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imwrite(str(out_dir / "gauss_5.png"), gauss)
    
    # 3. Median blur
    median = cv2.medianBlur(gray, 5)
    cv2.imwrite(str(out_dir / "median_5.png"), median)

    # Let's import skimage sauvola
    from skimage.filters import threshold_sauvola

    # Try Sauvola on gray directly (no CLAHE)
    for ws in [51, 101]:
        for k in [0.1, 0.15, 0.2]:
            # On raw gray
            thresh_raw = threshold_sauvola(gray, window_size=ws, k=k)
            bin_raw = (gray < thresh_raw).astype(np.uint8) * 255
            cv2.imwrite(str(out_dir / f"sauvola_raw_ws{ws}_k{k}.png"), bin_raw)
            
            # On Bilateral filtered
            thresh_bilat = threshold_sauvola(bilat, window_size=ws, k=k)
            bin_bilat = (bilat < thresh_bilat).astype(np.uint8) * 255
            cv2.imwrite(str(out_dir / f"sauvola_bilat_ws{ws}_k{k}.png"), bin_bilat)

            # On Gaussian filtered
            thresh_gauss = threshold_sauvola(gauss, window_size=ws, k=k)
            bin_gauss = (gauss < thresh_gauss).astype(np.uint8) * 255
            cv2.imwrite(str(out_dir / f"sauvola_gauss_ws{ws}_k{k}.png"), bin_gauss)

            # On Median filtered
            thresh_med = threshold_sauvola(median, window_size=ws, k=k)
            bin_med = (median < thresh_med).astype(np.uint8) * 255
            cv2.imwrite(str(out_dir / f"sauvola_median_ws{ws}_k{k}.png"), bin_med)

    print("no_clahe runs completed.")

if __name__ == "__main__":
    main()
