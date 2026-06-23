"""
rubbing_v4.py — Final binarisation for malayalam_stone/image15_original.jpeg (122x413)

Best approach found: upscale 3x + CLAHE + Otsu, then aggressive noise cleanup.
"""

import cv2
import numpy as np
from pathlib import Path
from skimage.filters import threshold_sauvola

INPUT = Path(
    r"C:\6th semester EL's\Interdisciplinary project\Implementation"
    r"\manuscript-digitisation\data\binarised_representative_samples"
    r"\malayalam_stone\image15_original.jpeg"
)
OUTPUT = INPUT.parent / "image15_rubbing_v4.png"


def binarise_malayalam_image15(img: np.ndarray) -> np.ndarray:
    """
    Binarisation for malayalam_stone/image15_original.jpeg.
    Dark rough stone slab, pale/white carved Malayalam characters.
    Output: white text on black background.

    Pipeline:
      3x upscale -> median (grain kill) -> CLAHE -> Otsu -> morph cleanup -> CCA
    """
    H, W = img.shape[:2]

    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Upscale 3x so filters operate above grain spatial frequency
    scale = 3
    large = cv2.resize(gray, (W * scale, H * scale), interpolation=cv2.INTER_CUBIC)
    LH, LW = large.shape

    # 3. Median blur (k=7) — kills grain without blurring character edges
    m = cv2.medianBlur(large, 7)

    # 4. CLAHE — normalise global luminance variance
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    e = clahe.apply(m)

    # 5. Otsu global threshold — works well when CLAHE has already normalised contrast
    _, b = cv2.threshold(e, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 6. Morphological opening — remove grain/texture specks (kernel=5 in upscaled space
    #    = ~1.7px in original; real character strokes are 4-8px in upscaled space)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((6, 6), np.uint8))

    # 7. Morphological closing — fill gaps in carved letter strokes
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # 8. Light dilation to thicken strokes
    b = cv2.dilate(b, np.ones((2, 2), np.uint8))

    # 9. CCA — remove extremely tiny dust AND extremely large stone-surface blobs
    n, labels, stats, _ = cv2.connectedComponentsWithStats(b, 8)
    canvas = np.zeros_like(b)
    min_area = 120         # in upscaled space: ~120/(3^2) ≈ 13 px² in original
    max_area = int(LH * LW * 0.06)  # < 6% of upscaled image per component

    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        canvas[labels == i] = 255

    # 10. Polarity — white chars on black
    if canvas.mean() >= 127:
        canvas = cv2.bitwise_not(canvas)

    # 11. Downscale back to original size
    result = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_AREA)
    _, result = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)

    return result


def main():
    img = cv2.imdecode(np.fromfile(str(INPUT), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"ERROR: Could not read {INPUT}")
        return
    print(f"Loaded: {img.shape}")
    result = binarise_malayalam_image15(img)
    white_pct = (result > 127).mean() * 100
    print(f"White %: {white_pct:.2f}%")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT), result)
    print(f"Saved -> {OUTPUT}")


if __name__ == "__main__":
    main()
