"""Enhanced binarisation test for malayalam_stone/image1_original.jpeg."""
from pathlib import Path
import cv2
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent))
from src.binarise import remove_noise_blobs

INPUT  = Path("data/binarised_representative_samples/malayalam_stone/image1_original.jpeg")
OUTDIR = Path("test_output/malayalam_binarise")
OUTDIR.mkdir(parents=True, exist_ok=True)

img = cv2.imdecode(np.fromfile(str(INPUT), dtype=np.uint8), cv2.IMREAD_COLOR)
assert img is not None

h, w = img.shape[:2]
shorter = min(h, w)


def binarise_palm_leaf_v2(img: np.ndarray) -> np.ndarray:
    """
    Enhanced palm-leaf binarisation.

    Key insight: for orange palm-leaf, the Red channel has the highest
    contrast between bright orange background (R≈220) and dark ink (R≈50).
    Steps:
      1. Extract R channel — best ink/background separation
      2. Bilateral filter — smooths fibre texture while preserving ink edges
      3. CLAHE — local contrast normalisation for uneven lighting
      4. Sauvola adaptive threshold — handles illumination gradients
      5. Flood-fill corners — removes border edge artefacts
      6. Morphological close — reconnects broken ink strokes
      7. Noise blob removal — kills isolated speckles
    """
    from skimage.filters import threshold_sauvola

    h, w = img.shape[:2]
    shorter = min(h, w)

    # Step 1: R channel (best orange-vs-ink contrast)
    r_channel = img[:, :, 2]

    # Step 2: bilateral filter — edge-preserving denoise
    # sigmaColor=30 keeps ink edges sharp; sigmaSpace scales with resolution
    sigma_s = max(5, shorter // 30)
    denoised = cv2.bilateralFilter(r_channel, d=9, sigmaColor=30, sigmaSpace=sigma_s)

    # Step 3: CLAHE for local contrast normalisation
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # Step 4: Sauvola — large window captures palm-leaf illumination gradient
    ws = max(21, min(61, (shorter // 8) | 1))   # odd, roughly 1/8 of shorter side
    if ws % 2 == 0:
        ws += 1
    k = 0.18   # moderate — orange bg is high-contrast, don't over-threshold
    thresh = threshold_sauvola(enhanced, window_size=ws, k=k)
    # Ink pixels have LOW R values → they are BELOW the threshold → white in INV
    binary = (enhanced < thresh).astype(np.uint8) * 255

    # Step 5: flood-fill corners to remove border noise
    mask = np.zeros((h + 2, w + 2), np.uint8)
    for corner in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
        cv2.floodFill(binary, mask, (corner[1], corner[0]), 0)

    # Step 6: morphological close — reconnect broken strokes (3×3 is safe)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # Step 7: remove noise blobs — min_size scales with image area
    min_size = max(15, (shorter // 120) ** 2)
    binary = remove_noise_blobs(binary, min_size=min_size, min_length=max(8, shorter // 80))

    # Polarity safety: white text on black background
    if binary.mean() >= 127:
        binary = cv2.bitwise_not(binary)

    return binary


result = binarise_palm_leaf_v2(img)
out = OUTDIR / "palm_leaf_v2.png"
cv2.imwrite(str(out), result)

white_pct = (result > 127).mean() * 100
print(f"palm_leaf_v2  white%={white_pct:.1f}  saved -> {out}")
print("Open: http://127.0.0.1:8000/test_output/malayalam_binarise/index.html")
