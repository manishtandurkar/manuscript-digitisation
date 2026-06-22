"""
Character-level segmentation binarisation.

Steps:
  1. Produce a rough binary mask (palm-leaf v2) to locate character regions.
  2. Dilate mask to merge nearby strokes into whole-character blobs.
  3. Find connected components — each is one character (or ligature).
  4. For each component: crop the same region from the ORIGINAL colour image,
     apply a tight local threshold, stamp the result onto a black canvas.
"""
from pathlib import Path
import cv2
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent))
from src.binarise import binarise_palm_leaf, remove_noise_blobs

INPUT  = Path("data/binarised_representative_samples/malayalam_stone/image1_original.jpeg")
OUTDIR = Path("test_output/malayalam_binarise")
OUTDIR.mkdir(parents=True, exist_ok=True)

original = cv2.imdecode(np.fromfile(str(INPUT), dtype=np.uint8), cv2.IMREAD_COLOR)
assert original is not None
H, W = original.shape[:2]

# ── Step 1: rough binary mask to locate characters ──────────────────────────
mask = binarise_palm_leaf(original)

# ── Step 2: dilate to merge nearby strokes (thin strokes split one character) ─
# kernel ~1/80 of shorter side — merges within-character gaps, not between chars
shorter = min(H, W)
dil_k = max(3, shorter // 40)
dilated = cv2.dilate(mask, np.ones((dil_k, dil_k), np.uint8))

# ── Step 3: connected components on dilated mask ─────────────────────────────
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)

# minimum character area: roughly (shorter/60)^2
min_area = max(20, (shorter // 60) ** 2)
# maximum area: anything bigger than ~40% of image is background noise
max_area = int(H * W * 0.40)

canvas = np.zeros((H, W), dtype=np.uint8)

pad = max(2, shorter // 80)   # small padding around each crop

for label in range(1, num_labels):
    area = int(stats[label, cv2.CC_STAT_AREA])
    if area < min_area or area > max_area:
        continue

    x  = int(stats[label, cv2.CC_STAT_LEFT])
    y  = int(stats[label, cv2.CC_STAT_TOP])
    cw = int(stats[label, cv2.CC_STAT_WIDTH])
    ch = int(stats[label, cv2.CC_STAT_HEIGHT])

    # padded crop bounds (clamped to image)
    x0 = max(0, x - pad);  y0 = max(0, y - pad)
    x1 = min(W, x + cw + pad); y1 = min(H, y + ch + pad)

    crop_orig = original[y0:y1, x0:x1]
    if crop_orig.size == 0:
        continue

    # ── Step 4: local binarisation on the crop ──────────────────────────────
    # R-channel: best orange-vs-ink contrast on palm leaf
    r = crop_orig[:, :, 2]

    # For very small crops fall back to Otsu; for larger use local Sauvola
    cr_h, cr_w = r.shape
    if cr_h < 10 or cr_w < 10:
        _, local_bin = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        from skimage.filters import threshold_sauvola
        ws = max(7, min(31, (min(cr_h, cr_w) // 3) | 1))
        if ws % 2 == 0:
            ws += 1
        thresh = threshold_sauvola(r, window_size=ws, k=0.15)
        local_bin = (r < thresh).astype(np.uint8) * 255

    # Stamp onto canvas (union: keep any white pixel from local binarisation)
    canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], local_bin)

# ── Step 5: final cleanup ────────────────────────────────────────────────────
# Light close to heal any gaps introduced by padding boundaries
canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

# Remove tiny noise blobs that sneak through
canvas = remove_noise_blobs(canvas, min_size=max(10, (shorter // 150) ** 2),
                            min_length=max(5, shorter // 100))

# Polarity: must be white text on black
if canvas.mean() >= 127:
    canvas = cv2.bitwise_not(canvas)

out = OUTDIR / "segmented_letters.png"
cv2.imwrite(str(out), canvas)
white_pct = (canvas > 127).mean() * 100
print(f"segmented_letters  white%={white_pct:.1f}  components={num_labels-1}  -> {out}")
