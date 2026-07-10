# Technical Explanation: Inscription Digitisation Project Implementation

**Date:** July 10, 2026
**Phase:** Phase 1 complete, Phase 2 (Stages 4–7) implemented — only machine translation remains unbuilt
**Status:** Full seven-stage pipeline (preprocess → enhance → binarise → OCR → metrics → record assembly → orchestration) implemented with automated testing, a FastAPI backend, and a React web UI

---

## 1. Project Context & Objectives

The Inscription Digitisation Project transforms degraded scanned images of historical South Asian artefacts — stone inscriptions, rubbings/estampages, palm leaf manuscripts, copper/metal plates, and cave/rock paintings — into high-quality, machine-readable research records.

The core mission is to extract legible text from unclear source images through a multi-stage image processing pipeline, then transcribe, evaluate, and package the result as a structured research record. What was originally planned as "Phase 1 (image processing) then Phase 2 (OCR onward)" has since collapsed into one working end-to-end pipeline: OCR, quality metrics, record assembly, and orchestration are all implemented. Only Stage 5 (machine translation to English) remains a placeholder.

### Pipeline Overview

```
Raw Image (JPG/PNG/TIF/AVIF/WEBP)
    ↓
[Stage 1: Preprocessing]   — Normalise the scan itself (fix orientation, exposure, colour, borders)
    ↓
[Stage 2: Enhancement]     — Improve legibility (AI super-resolution, denoising, pigment reveal) — now auto-routed by document type
    ↓
[Stage 3: Binarisation]    — Convert to strict black/white for OCR — document-type-aware, with adaptive parameter tuning
    ↓
[Stage 4: OCR]             — Tesseract + EasyOCR ensemble transcription — IMPLEMENTED
    ↓
[Stage 5: Translation]     — Convert to English — still a placeholder (src/record.py emits "phase_2_pending")
    ↓
[Stage 6: Record Assembly] — Bundle all outputs + quality metrics into structured JSON, with optional PDF export — IMPLEMENTED
    ↓
[Stage 7: Orchestration]   — Single-image and parallel batch pipeline runner — IMPLEMENTED
```

---

## 2. The Crucial Distinction: Preprocessing vs Enhancement

This distinction is frequently misunderstood and is the conceptual centrepiece of the pipeline. It has not changed since the original design.

### Preprocessing — "Fix the scan"

Preprocessing corrects **problems introduced by the scanning or photography process itself**. It does not add information; it removes distortions.

- The artefact was photographed under inconsistent lighting → CLAHE fixes local exposure.
- The camera stored an incorrect rotation in EXIF → `exif_transpose` corrects it.
- The image has a colour cast from artificial light → white balance corrects it.
- The scanner introduced a black border → border crop removes it.

**Analogy:** Preprocessing is like adjusting the camera settings after the shot. The underlying information was always there; you are removing a layer of distortion that obscures it.

### Enhancement — "Recover lost information"

Enhancement applies AI and signal processing to **improve the legibility of the artefact itself**. It actively synthesises or reveals information.

- Character strokes are blurry from a low-resolution camera → Real-ESRGAN synthesises sharper detail.
- Noise (dust, scanner grain) obscures text → Non-Local Means denoising suppresses it.
- Cave pigment has faded to near-invisible colour differences → DStretch decorrelation stretch amplifies them.
- Denoised strokes have soft edges → Unsharp mask sharpens them.

**Analogy:** Enhancement is like a forensic photograph expert enhancing a blurry image — it goes beyond the raw capture to recover what the raw scan could not faithfully represent.

### Side-by-Side Comparison

| Property | Preprocessing | Enhancement |
|---|---|---|
| Goal | Remove scan artefacts | Improve legibility of the artefact |
| Information change | Removes distortion, no net gain | Synthesises/reveals new detail |
| Algorithms | CLAHE, grey-world AWB, crop | Real-ESRGAN, NLM denoise, DStretch |
| AI involved? | No (classical signal processing) | Yes (deep learning for super-resolution) |
| Changes resolution? | No (may reduce via crop) | Yes when `mode="superres"` (2× upscale) |
| Speed | 1–2 seconds | 1–25 seconds depending on mode |
| Order in pipeline | Must be first | Requires preprocessed input |

---

## 3. Stage 1: Preprocessing — Technical Deep Dive

**Source:** [src/preprocess.py](../src/preprocess.py) — unchanged since the original design.

The preprocessing chain executes four operations in strict sequence:

```
load_image()  →  normalise_brightness()  →  auto_white_balance()  →  crop_borders()
```

### 3.1 Image Loading with EXIF Correction

```python
def load_image(path: str) -> np.ndarray:
    with PilImage.open(path) as pil_img:
        pil_img = ImageOps.exif_transpose(pil_img)
        rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
```

- Opens via PIL (not OpenCV directly) to access EXIF metadata before pixel decoding.
- `ImageOps.exif_transpose()` reads the EXIF Orientation tag (values 1–8) and physically rotates/flips pixel data to match. This is critical: smartphones commonly store images rotated 90° with an EXIF correction tag rather than rotating the pixel data.
- Converts PIL RGB → OpenCV BGR for downstream compatibility with `cv2.*` functions.

**Why PIL before OpenCV?** OpenCV's `imread()` ignores EXIF orientation. Without correction, all subsequent algorithms (cropping, binarisation) would operate on a rotated image.

### 3.2 Brightness Normalisation via CLAHE

```python
def normalise_brightness(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
```

**CLAHE = Contrast Limited Adaptive Histogram Equalisation.**

1. Converts BGR → LAB colour space. LAB separates luminance (L channel) from colour (A = green-red axis, B = blue-yellow axis). This matters because histogram equalisation should only touch brightness, not hue.
2. Applies CLAHE to the L channel only (A and B are untouched — colours do not shift).
3. `tileGridSize=(8,8)` divides the image into an 8×8 grid of tiles, equalising each independently, making it *adaptive* — a dark shadow in one corner does not affect equalisation of a bright centre.
4. `clipLimit=2.0` caps the histogram gradient to prevent noise amplification.
5. Merges and converts back to BGR.

**Why not global histogram equalisation?** A stone inscription photographed outdoors has sunlit areas and deeply shadowed crevices in the same frame. Global HE would normalise across both, washing out the detail in shadows or blowing out the highlights. CLAHE adapts locally, preserving both.

### 3.3 Grey-World White Balance

```python
def auto_white_balance(img: np.ndarray) -> np.ndarray:
    img_float = img.astype(np.float32)
    channel_means = img_float.reshape(-1, 3).mean(axis=0)
    overall_mean = float(channel_means.mean())
    scale = overall_mean / np.maximum(channel_means, 1e-6)
    balanced = img_float * scale.reshape(1, 1, 3)
    return np.clip(balanced, 0, 255).astype(np.uint8)
```

**Grey-world assumption:** the average colour across a natural image should be neutral grey. Any deviation from grey is a colour cast from the light source. The function scales each channel so its mean equals the overall mean, then clips to `[0, 255]`.

**Limitation:** fails if a scene is dominated by a single colour (e.g., a very green moss-covered stone). No alternative AWB has been added.

### 3.4 Border Cropping

```python
def _crop_borders_with_metadata(img, threshold=10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = ((gray > threshold) & (gray < 255 - threshold)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    points = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(points)
    if w < width * 0.25 or h < height * 0.25:
        return img.copy(), (0, 0, width, height)
    cropped = img[y : y + h, x : x + w]
    return cropped, (x, y, w, h)
```

1. Creates a binary mask: pixels with values in the interior range `(10, 245)` are content; near-black and near-white are border.
2. `MORPH_CLOSE` connects fragmented content regions.
3. `cv2.boundingRect()` computes the tightest bounding box around all non-zero mask pixels.
4. **Sanity check:** if the detected content box is under 25% of the original width or height, the crop is rejected and the original returned unchanged — this prevents aggressive over-cropping when borders are misdetected.
5. Returns both the cropped image and crop coordinates `(x, y, w, h)` for logging.

**Output:** JPEG at quality 95 saved to `data/preprocessed/`.

---

## 4. Stage 2: Enhancement — Technical Deep Dive

**Source:** [src/enhance.py](../src/enhance.py)

The biggest change since Phase 1 is that `enhance()` is now **document-type-aware and self-routing** rather than requiring the caller to pick a mode. It calls `detect_document_type()` from `src/binarise.py` to decide.

```python
def enhance(img_path, output_path, use_dstretch=False, mode="auto"):
    doc_type = detect_document_type(img, img_path=img_path)
    is_already_high_contrast = (
        doc_type == "stone"
        and cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).std() < 30
        and cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1].mean() < 20
    )
    if mode == "auto":
        if use_dstretch or doc_type == "palm_leaf":
            mode = "dstretch"
        elif is_already_high_contrast:
            mode = "mild"
        elif min(h, w) < 500:
            mode = "superres"
        else:
            mode = "mild"

    img = denoise(img, strength=8 if doc_type == "palm_leaf" else 10)
    if mode == "dstretch":
        img = dstretch(img)
    elif mode == "superres":
        img = enhance_with_realesrgan(img)
    # "mild": denoise + sharpen only

    sharpen_amount = 1.0 if doc_type == "palm_leaf" else 1.5
    img = sharpen(img, amount=sharpen_amount)
```

**Auto-routing rules, in order:**
- `use_dstretch=True` or palm-leaf document type → `dstretch` (reveals faded ink/pigment channels).
- Stone image that is already high-contrast (low std-dev, low saturation — i.e. already crisp, near-achromatic) → `mild` (denoise + sharpen only; super-resolution would over-smooth an already-good image).
- Low-resolution stone (shorter side < 500px) → `superres` (Real-ESRGAN, since detail genuinely needs to be synthesised).
- Everything else (normal-resolution stone/rubbings) → `mild`.

This is a deliberate shift from the original design, where Real-ESRGAN 2× super-resolution ran by default on every stone image. In practice most stone photographs are already high enough resolution that `mild` (denoise + sharpen) gives a cleaner result for binarisation than upscaling does; `superres` is now reserved for genuinely small/low-res captures.

### 4.1 Non-Local Means Denoising

```python
def denoise(img: np.ndarray, strength: int = 10) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)
```

`cv2.fastNlMeansDenoisingColored()` is a colour-aware implementation of Non-Local Means (NLM) denoising. For each target pixel, it searches a large window (21×21px) for similar patches (7×7px), and each found patch votes on the target pixel's value, weighted by patch similarity rather than mere spatial proximity — unlike Gaussian blur.

`strength=10` is the default; palm leaf images use `strength=8` (milder, since thin ink strokes are easy to erode) while everything else uses `strength=10`.

### 4.2 Real-ESRGAN Super-Resolution

```python
def _build_upsampler(model_path: str):
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4)
    return RealESRGANer(scale=4, model_path=model_path, model=model,
                        tile=400, tile_pad=10, pre_pad=0, half=False)

def enhance_with_realesrgan(img, scale=2, model_path=DEFAULT_MODEL_PATH):
    upsampler = _get_upsampler(str(mp.resolve()))
    output_rgb, _ = upsampler.enhance(img_rgb, outscale=scale)
    return cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR).astype(np.uint8)
```

Unchanged from the original design: RRDBNet (Residual-in-Residual Dense Block Network) generator, 4× trained model run at `outscale=2` (avoids over-smoothing character strokes), 400px tiling with 10px overlap to bound memory use, `half=False` for full float32 precision, and an `@lru_cache(maxsize=2)` on the built upsampler to avoid the 2–3s reload cost per call. Weights auto-download from the official GitHub release URL on first use. `models/weights/` now also holds `RealESRGAN_x4plus_anime_6B.pth` alongside the base model (unused by the current pipeline — an artefact of experimentation).

One implementation addition: `_build_upsampler()` now patches `sys.modules['torchvision.transforms.functional_tensor']` before importing BasicSR, working around a BasicSR/newer-torchvision incompatibility where that module was removed.

### 4.3 DStretch Decorrelation Stretch

Unchanged. Used for cave paintings and — newly — for **all palm leaf manuscripts** under the `mode="auto"` routing above (see §4), since decorrelation stretch helps separate faded ink from tan fibre background as well as faded pigment from rock.

```python
def dstretch(img: np.ndarray, colour_space: str = "LAB") -> np.ndarray:
    ...
    cov = (centered.T @ centered) / (n - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    stretch_matrix = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues + 1e-10)) @ eigenvectors.T
    stretched = centered @ stretch_matrix
    ...
```

Eigenvalue decomposition of the RGB covariance matrix identifies the principal colour axes; the stretch matrix equalises variance along each axis, amplifying subtle colour differences (faded pigment, or faded ink vs. tan leaf fibre) that are invisible in the raw capture. One important operational rule downstream: **DStretch output is never used as binarisation input** (see §5.9) — its recolouring breaks binarisation thresholds calibrated on natural stone/leaf tones.

### 4.4 Unsharp Mask Sharpening

Unchanged:

```python
def sharpen(img: np.ndarray, amount: float = 1.5) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)
```

`amount=1.5` for most images, `amount=1.0` for palm leaf (thin strokes break under aggressive sharpening).

---

## 5. Stage 3: Binarisation — Technical Deep Dive

**Source:** [src/binarise.py](../src/binarise.py) — now 1,600+ lines, the largest and most-evolved module in the codebase.

Binarisation converts a colour or grayscale image to strictly binary (0 or 255) — a prerequisite for OCR. Output convention is now **white text on black background** (`255` = ink/carving, `0` = background) for every method, a change from the earlier "0 = background (white), 255 = foreground (black)" PNG convention — this matches what Tesseract/EasyOCR and the connected-component noise-removal logic expect more naturally.

### 5.1 Document-Type Detection

```python
def detect_document_type(img: np.ndarray, img_path=None) -> str:
    if img_path is not None:
        path_str = str(img_path).lower()
        if "stone" in path_str or "rubbing" in path_str or "vijay" in path_str:
            return "stone"
        if "palm" in path_str or "leaf" in path_str:
            return "palm_leaf"
        if "metal" in path_str or "copper" in path_str or "plate" in path_str:
            return "metal_plate"

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_hue = float(hsv[:, :, 0].mean())
    mean_sat = float(hsv[:, :, 1].mean())
    aspect_ratio = max(w / h, h / w)
    mean_corner_val = np.mean([corner.mean() for corner in four_5x5_corners])

    if mean_sat > 75 and 8 <= mean_hue <= 30 and aspect_ratio > 1.8 and mean_corner_val < 200:
        return "palm_leaf"

    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var > 25000:
        return "metal_plate"

    return "stone"
}
```

Now detects **three** document types (was two): `stone`, `palm_leaf`, `metal_plate`. Two changes from the original heuristic:

- A **path-hint fast path** checks the image's file path for keywords (`stone`, `rubbing`, `palm`, `metal`, `copper`, `plate`) before falling back to pixel analysis — a pragmatic addition since the dataset's folder naming is a highly reliable signal and pixel heuristics occasionally misfire on edge cases.
- The palm-leaf saturation threshold was raised from 40 to 75 and now also requires a long/narrow aspect ratio (`>1.8`) and dark image corners (`<200`), to avoid misclassifying reddish sandstone as palm leaf.
- `metal_plate` detection is new: engraved copper/bronze plates are near-achromatic but have an extremely high-frequency grayscale signature (Laplacian variance > 25000) from punched/engraved strokes against smooth patina.

A related helper, `detect_rubbing()`, separately flags rubbing/estampage-style stone images (chalk-on-paper impressions) via moderate-to-high saturation, high local speckle texture, and strong global contrast — used by the top-level `binarise()` dispatcher to route to a dedicated rubbing path (§5.6) before the general stone path runs.

### 5.2 Stone Binarisation — `binarise_stone()`

The original black-hat morphological approach has been **replaced**. The current method is bilateral filtering + adaptive Sauvola local thresholding, with a separate code path for high-resolution images:

```python
def binarise_stone(img: np.ndarray) -> np.ndarray:
    gray = _to_gray(img)
    shorter = min(H, W)

    if shorter >= 1500:
        # High-res branch: median blur (kernel ~ shorter/100) + Sauvola
        # (window ~ shorter/30, k=0.25), then noise removal + border flood-fill.
        ...
        return binary

    # Standard branch (<1500px shorter side):
    d = 5 if shorter < 500 else 9
    denoised = cv2.bilateralFilter(gray, d=d, sigmaColor=sigma, sigmaSpace=sigma)

    ws = max(15, min(151, (shorter // 12) | 1))
    k = 0.12 if shorter < 500 else 0.15
    thresh = threshold_sauvola(denoised, window_size=ws, k=k)
    binary = (denoised < thresh).astype(np.uint8) * 255

    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, ...)
    binary = remove_noise_blobs(binary, min_size=..., min_length=...)
    if binary.mean() >= 127:
        binary = cv2.bitwise_not(binary)   # polarity safety check
    # flood-fill from every border pixel to strip outer margin noise
    return binary
```

Key design points:

- **Bilateral filter, not Gaussian blur.** Bilateral filtering smooths stone grain texture while preserving edge sharpness at carved-stroke boundaries — Gaussian blur would soften both equally.
- **All Sauvola window/`k` parameters scale with image size** (`shorter // 12`, clamped `[15, 151]`), rather than using one fixed window as in the original design. Small images get a tight window; large images get a wide one.
- **Separate high-resolution branch** (shorter side ≥ 1500px): uses median blur (better than bilateral at killing rough granite grain at this scale) and a wider, more aggressive Sauvola window/`k` combination, calibrated for very large outdoor photographs.
- **Polarity safety check:** after thresholding, if more than half the image is white the image is assumed inverted and is flipped — Sauvola's `(gray < thresh)` convention can invert depending on lighting.
- **Border flood-fill:** after thresholding, `cv2.floodFill` is run from every pixel along the four image edges, removing any white regions connected to the border (residual scan margin noise) without touching interior text.

Both branches call a **new adaptive parameter helper**, `_adaptive_sauvola_params()`, when `binarise_sauvola()` (the generic/non-stone-specific method) is used directly:

```python
def _adaptive_sauvola_params(gray: np.ndarray) -> tuple[int, float]:
    # Base window ~= shorter/20, clamped [15, 71]; widened further for
    # high-res images (>1500px: +10, >800px: +6).
    # Base k=0.20; raised to 0.30 for low-contrast (std<25) images,
    # to 0.25 for moderately low-contrast (std<40); floored at 0.18 for
    # very dark (mean<60, rubbing-like) images; capped at 0.15 for very
    # bright (mean>160, pale outdoor stone) images to catch faint strokes.
```

This tunes `window_size` and `k` from the image's own brightness/contrast statistics rather than using one hand-picked constant for every image — a direct response to the dataset's wide variation in lighting and stone colour.

### 5.3 Palm Leaf Binarisation — Character-Level Segmentation

The palm-leaf path was **rebuilt from a single adaptive-threshold pass into a two-stage, per-character segmentation pipeline** — the biggest algorithmic change in the module:

```python
def binarise_palm_leaf(img: np.ndarray) -> np.ndarray:
    # 1. Rough mask: R-channel + bilateral filter + CLAHE + Sauvola,
    #    corner flood-fill to drop background.
    rough = _palm_leaf_rough_mask(img)
    # 2. Dilate to merge intra-character stroke fragments into solid blobs.
    dilated = cv2.dilate(rough, ...)
    # 3. Connected components -> one blob per character/ligature cluster.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    for label in range(1, num_labels):
        # skip components outside [min_area, max_area]
        crop = img[y0:y1, x0:x1]                    # original-colour crop
        r_crop = crop[:, :, 2]                       # R channel of that crop only
        thresh = threshold_sauvola(r_crop, window_size=ws, k=0.15)
        local_bin = (r_crop < thresh).astype(np.uint8) * 255
        # 5. Stamp this locally-binarised crop onto the output canvas.
        canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], local_bin)
```

**Why per-character local thresholding?** A single global (or even locally-adaptive-window) threshold applied to the whole leaf still has to cope with illumination gradients across the leaf's length. By first locating each character-sized blob with a rough mask, then re-running Sauvola thresholding *inside just that blob's bounding box*, each local threshold only has to handle near-uniform lighting — eliminating the background bleed that a whole-image pass leaves behind. The rough mask (`_palm_leaf_rough_mask`) exists purely to find *where* the characters are; its own output is discarded once segmentation is done, and the final pixel values always come from re-thresholding the original colour crop.

Final cleanup: `MORPH_CLOSE`, then `remove_noise_blobs()` with palm-leaf-specific thresholds, then a polarity check.

### 5.4 Metal / Copper Plate Binarisation — New Methods

Two new dedicated methods handle engraved metal artefacts, which behave nothing like stone or leaf:

- **`binarise_metal_plate()`** — for plates where engraved strokes are a starkly different material against a dark, low-texture patina: a single global Otsu threshold is sufficient, followed by small-blob connected-component cleanup.
- **`binarise_copper_plate()`** — for horizontal copper plates photographed with light text on a dark plate surrounded by a bright border/mount. This method first isolates the plate's rectangular bounding box by thresholding against the bright surrounding background, crops 6px inward to exclude the border line, then runs Sauvola thresholding **inverted** (`gray > thresh`, since text is *lighter* than the plate here) followed by the same character-level connected-component segmentation approach used for palm leaf.

The top-level `binarise()` dispatcher detects copper-plate images heuristically (aspect ratio > 1.8, bright corners, dark overall mean) independently of `detect_document_type()`, and routes to `binarise_copper_plate()` before the general `sauvola`/`otsu`/`adaptive` dispatch runs.

### 5.5 Otsu, Adaptive Mean, U-Net, DocEnTr — Largely Unchanged

`binarise_otsu()` and `binarise_adaptive()` are functionally the same as before (Otsu global / adaptive-mean local thresholding after CLAHE), with output polarity now inverted to match the "white text, black background" convention (`THRESH_BINARY_INV`).

The two deep-learning methods, `binarise_unet()` (Lightweight U-Net, encoder-decoder with skip connections) and `binarise_docentr()` (patch-ViT transformer encoder + CNN decoder, based on El-Hajj & Barakat, ArXiv 2209.09921), are unchanged in architecture. Both still call `_dl_infer()` and fall back to `binarise_sauvola()` when confidence (`_binary_entropy_confidence`, mean `1 − pixel entropy`) is below `_CONFIDENCE_THRESHOLD = 0.65`, or when no weights file exists at `models/weights/unet_binarise.pth` / `docentr_binarise.pth`. No trained weights currently ship with the repo, so both methods are, in practice, always Sauvola fallbacks unless weights are supplied externally.

### 5.6 Rubbing/Estampage Path — New

```python
def binarise_rubbing(img: np.ndarray) -> np.ndarray:
    """Median-blur(13) + Otsu only. Parameters determined empirically."""
    median = cv2.medianBlur(gray, 13)
    _, binary = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ...)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, ...)
```

Rubbings (chalk/ink impressions taken directly off carved stone onto paper) already have strong, near-binary local contrast — Sauvola's adaptive windowing is unnecessary and can introduce noise from paper grain. A simple median-blur + Otsu global threshold is deliberately kept minimal here.

### 5.7 Per-Image Calibration Overrides — A Pragmatic (and Fragile) Layer

The `binarise()` dispatcher contains roughly a dozen **hard-coded routing rules** that send specific, individually-troublesome source images (matched by filename substring, or by exact pixel dimensions as a fallback) to their own bespoke binarisation function — e.g. `binarise_image1()`, `binarise_image3()`, `binarise_malayalam_image9()`, `binarise_malayalam_image15()`, `binarise_tamil_010()`, `binarise_tamil_026()`, `binarise_img3924()`. Each of these was tuned interactively against one specific problem image during development (extreme upscaling, custom ROI crops, hand-picked connected-component area/aspect-ratio filters, or channel selection) and several load the *raw* source image directly from `data/raw/` rather than trusting the pipeline's preprocessed/enhanced intermediate, because the general enhancement chain produced worse results for that specific photograph.

This is useful for getting a clean result on the exact images used to develop the pipeline, but is a known **generalisation gap**: it does not scale past the handful of images it was written for, and dimension-based matching (e.g. `img.shape[0] == 156 and img.shape[1] == 323`) is brittle — a coincidentally same-sized new image would be silently misrouted. See §12 (Known Limitations).

### 5.8 Noise Removal

`remove_noise_blobs()` is unchanged in logic (keep a connected component if `area >= min_size` **OR** `max(width, height) >= min_length`, using `cv2.connectedComponentsWithStats`), but `min_size`/`min_length` are now auto-scaled from image size (`shorter // 200`, floored) when not explicitly passed, rather than requiring the caller to pick fixed numbers.

### 5.9 DStretch-Enhanced Images Are Never Binarisation Input

`api/pipeline.py`'s `_run_binarise()` explicitly picks only `*_enhanced_superres*.jpg` outputs as binarisation source when they exist (falling back to preprocessed, then raw) — **never** a `dstretch`-mode enhancement output, because DStretch's colour-channel recolouring distorts the pixel statistics that Sauvola/Otsu thresholding depend on. Palm-leaf images bypass the enhanced/preprocessed intermediates entirely and binarise straight from the raw image, since the character-segmentation approach (§5.3) already does its own local contrast normalisation.

### 5.10 Binarisation Method Selection Guide (Current)

| Document Type | Routing | Reason |
|---|---|---|
| Stone inscription | `binarise_stone` (bilateral/median + adaptive Sauvola) | Handles grain vs. carved-groove separation across a wide range of resolutions |
| Rubbing/estampage | `binarise_rubbing` (median + Otsu) | Already near-binary; adaptive windowing adds noise |
| Palm leaf manuscript | `binarise_palm_leaf` (character-level segmentation) | Per-character local thresholding eliminates background bleed from illumination gradients |
| Metal/copper plate | `binarise_metal_plate` / `binarise_copper_plate` | Different material contrast model entirely (engraved vs. carved) |
| Clean paper manuscript | `otsu` | Fast, accurate on uniform backgrounds |
| Mixed/uncertain | `adaptive` | Safe fallback |
| DL weights available | `unet` or `docentr` | Highest accuracy if trained weights are supplied (none ship currently) |
| Specific known-problem images | Dedicated per-image function | Interactively tuned; not intended to generalise (§5.7) |

**Output:** 8-bit single-channel PNG, white text (255) on black background (0) — Tesseract/EasyOCR-ready.

---

## 6. Stage 4: OCR & Transcription — Now Implemented

**Source:** [src/ocr.py](../src/ocr.py)

Dual-engine ensemble, matching the original design intent:

### 6.1 Script Configuration & Detection

```python
SCRIPT_CONFIG = {
    "tamil":      {"tesseract_lang": "tam", "easyocr_lang": ["ta"]},
    "sanskrit":   {"tesseract_lang": "san", "easyocr_lang": ["hi"]},
    "kannada":    {"tesseract_lang": "kan", "easyocr_lang": ["kn"]},
    "telugu":     {"tesseract_lang": "tel", "easyocr_lang": ["te"]},
    "malayalam":  {"tesseract_lang": "mal", "easyocr_lang": ["ml"]},
    "devanagari": {"tesseract_lang": "hin", "easyocr_lang": ["hi"]},
    "brahmi":     {"tesseract_lang": None,  "easyocr_lang": None},
    "grantha":    {"tesseract_lang": None,  "easyocr_lang": None},
}
```

`detect_script()` uses connected-component geometry as a lightweight heuristic: it Otsu-binarises the image, measures the median aspect ratio and area of connected components, and flags Devanagari when components are tall/narrow with a distinctive top bar signature (`median_aspect < 0.7`); otherwise it defaults to Tamil, the project's primary test script. This is a coarse heuristic, not a trained script classifier — real multi-script robustness would need one.

Brahmi and Grantha have no configured OCR engine (`tesseract_lang`/`easyocr_lang` both `None`); `transcribe()` detects this up front and returns a `status: "manual_transcription_required"` result without attempting OCR — matching the original Phase 2 plan to defer these scripts to a fine-tuned model.

### 6.2 Tesseract Configuration

```python
_TESSDATA_DIR = Path(__file__).resolve().parents[1] / "tessdata"
_TESS_BINARY  = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
_TESS_CONFIG = "--oem 1 --psm 6"

if _TESS_BINARY.exists():
    pytesseract.pytesseract.tesseract_cmd = str(_TESS_BINARY)
if _TESSDATA_DIR.exists():
    os.environ["TESSDATA_PREFIX"] = str(_TESSDATA_DIR)
```

- Tesseract binary path and `TESSDATA_PREFIX` are auto-configured at import time if a local Windows Tesseract install and a bundled `tessdata/` directory exist — no manual environment setup needed on the target development machine.
- `tessdata/` in the repo root bundles `eng.traineddata`, `tam.traineddata`, `kan.traineddata`, `tel.traineddata`, `mal.traineddata` — the four South Indian scripts plus English, checked into the project rather than relying on system-wide Tesseract language packs.
- `--oem 1` selects the LSTM recognition engine; `--psm 6` assumes a single uniform block of text (appropriate for cropped/binarised inscription regions).
- `ocr_tesseract()` calls `pytesseract.image_to_data()` (not `image_to_string()`) specifically to get per-word bounding boxes and confidence scores, not just a flat text block.

### 6.3 EasyOCR

`ocr_easyocr()` lazily constructs and caches a `easyocr.Reader(langs, gpu=False, verbose=False)` per unique language-list key (`_EASYOCR_READER_CACHE`), since Reader construction loads a deep model and is expensive. Both engines return the same schema — `{"text", "confidence", "word_boxes": [{"text", "confidence", "box": [x,y,w,h]}]}` — so the ensemble logic can treat them uniformly.

### 6.4 Ensemble Merge

```python
def ocr_ensemble(img, script):
    tess = ocr_tesseract(img, tess_lang)
    easy = ocr_easyocr(img, easy_langs)
    if tess["confidence"] >= easy["confidence"]:
        primary, secondary = tess, easy
    else:
        primary, secondary = easy, tess
    # merge word boxes: primary's boxes + any secondary box at a position
    # the primary engine didn't already report
```

The higher mean-confidence engine's full text becomes the primary transcription; the other engine's word boxes are folded in wherever they occupy a position the primary engine missed, giving denser bounding-box coverage for downstream uncertain-region flagging. `engine_used` is reported as `"tesseract"`, `"easyocr"`, or `"tesseract+easyocr ensemble"` depending on whether both contributed.

### 6.5 Line Grouping

`_group_words_into_lines()` sorts word boxes by `(top-y, left-x)` and greedily groups words into a line whenever a word's top-y is within `max(current_line_height * 0.6, 8)` pixels of the running line's y — a simple but effective heuristic for OCR output that arrives as an unordered bag of word boxes rather than pre-segmented lines. Each resulting line carries its own mean confidence and is flagged `uncertain` if that mean falls below the review threshold.

### 6.6 Confidence Tiers (matches the original Phase 2 design exactly)

```python
_CONFIDENCE_VERIFIED = 0.85
_CONFIDENCE_REVIEW   = 0.60
```

- `>= 0.85` → `"verified"` (auto-accept)
- `0.60–0.84` → `"review_needed"`
- `< 0.60` → `"uncertain"`

`overall_confidence` is the mean of per-line confidences (falling back to the ensemble's raw confidence if no lines were grouped, or `0.0` if nothing was recognised at all). Word boxes below the review threshold are also collected into `uncertain_regions` — a list of `[x1, y1, x2, y2]` rectangles — independent of line-level flagging, so a UI could highlight specific low-confidence words even within an otherwise "verified" line.

### 6.7 Output Schema

```json
{
  "script": "tamil",
  "text": "...",
  "lines": [{"line_number": 1, "text": "...", "confidence": 0.91, "bounding_box": [...], "uncertain": false}],
  "overall_confidence": 0.87,
  "confidence_status": "verified",
  "engine_used": "tesseract+easyocr ensemble",
  "uncertain_regions": [[x1,y1,x2,y2], ...],
  "duration_s": 1.23
}
```

Both engines "degrade gracefully" per the module docstring: if `pytesseract`/`easyocr` are not importable, or if the underlying engine call throws, an empty result (`confidence=0.0`, empty text/boxes) is returned rather than raising — so a transcription request never crashes the pipeline even on a machine without both OCR backends installed.

---

## 7. Stage 5: Translation — Still a Placeholder

Unlike OCR, translation has **not** been implemented. `src/record.py`'s `assemble_record()` accepts an optional `translation` dict; when `None` (the only case currently reachable — nothing in the pipeline produces a translation), it fills in:

```python
{"english": None, "modern_source_language": None, "confidence": None,
 "method": None, "notes": [], "status": "phase_2_pending"}
```

The original plan (Helsinki-NLP OPUS-MT for post-10th-century texts, Claude/GPT-4 API fallback for classical/archaic forms) remains the intended design but has no corresponding `src/translate.py` module yet.

---

## 8. Stage 6: Record Assembly — Now Implemented

**Source:** [src/record.py](../src/record.py)

`assemble_record()` bundles preprocessing/enhancement/binarisation image paths, transcription output, the (currently placeholder) translation block, the quality report, and a processing log into one structured record dict, matching the schema the original design specified. Notable implementation details:

- **Record IDs** (`generate_record_id()`) are auto-sequenced per calendar year in the form `INS-YYYY-NNNN`, by scanning `data/records/` for the highest existing sequence number for the current year and incrementing — no external ID service or database needed for the current single-writer setup.
- **Status** is derived automatically from transcription confidence: `verified` (≥0.85), `review` (≥0.60), else `draft` — reusing the same thresholds as OCR confidence tiers.
- **Citation block** auto-generates a suggested citation string from location/period metadata and the current date, plus a fixed `CC BY 4.0` licence field and a `doi: null` placeholder for future DOI assignment.
- **`save_record()`** writes `data/records/{record_id}.json`.
- **`export_pdf()`** (requires the optional `fpdf2` dependency, raises `ImportError` with an install hint if missing) generates a researcher-facing PDF: artefact metadata table, side-by-side original/enhanced images (skipped gracefully if files are missing), transcription text with confidence, a "Phase 2 pending" translation placeholder, quality metrics, and the citation block.

---

## 9. Quality Metrics — Self-Referenced (Rewritten from the Original Design)

**Source:** [src/metrics.py](../src/metrics.py)

The original PSNR/SSIM design assumed a clean ground-truth image to compare against — but inscription images have none; there is no "correct" version of a 1,000-year-old carved stone to diff against. The metrics module was rewritten around **self-reference**:

```python
def _make_pseudo_reference(enhanced: np.ndarray) -> np.ndarray:
    """Mild bilateral filter as a pseudo ground-truth: removes high-frequency
    processing artefacts while keeping stroke structure."""
    return cv2.bilateralFilter(_to_grey(enhanced), d=9, sigmaColor=25, sigmaSpace=25)
```

`compute_psnr()` and `compute_ssim()` both compare the enhanced image against *its own* bilateral-smoothed version, not against the pre-enhancement original. This reframes the question from "how different is this from the raw scan" (not meaningful — enhancement is *supposed* to change the image) to "how free of ringing/over-sharpening/noise-amplification artefacts is the final output" (meaningful — a heavily over-processed image will diverge sharply from its own smoothed self). Target thresholds: PSNR ≥ 30 dB, SSIM ≥ 0.85.

Two metrics are unchanged in concept: `compute_cnr()` (contrast-to-noise ratio between an Otsu-derived text mask and background, target ≥ 1.5) and `compute_sharpness()` (Laplacian variance).

One new metric, **ink coverage** (`compute_ink_coverage()`), is a practical sanity check on the *binarised* output: percentage of white (text) pixels. `full_quality_report()` flags failure if coverage is below 0.5% (binarisation extracted essentially nothing) or above 45% (binarisation flooded with noise rather than isolating text). `passes_thresholds` is `True` only if PSNR, SSIM, CNR, and ink coverage (when available) all clear their thresholds.

---

## 10. Stage 7: Pipeline Orchestration — Now Implemented

**Source:** [src/pipeline.py](../src/pipeline.py)

`process_single(image_path, artefact_meta=None, script="auto", use_dstretch=False, binarise_method="sauvola", save_record=True, export_pdf=False)` runs all five processing stages sequentially — preprocess → enhance → binarise → OCR → record assembly — via an internal `_stage()` wrapper that times each step, catches exceptions per-stage (so one failing stage doesn't abort the whole record — it's logged and the pipeline falls back to the previous stage's output as input to the next), and accumulates a `processing_log` that ends up in the final record.

`process_batch(input_dir, pattern="*.jpg", meta_csv=None, script="auto", workers=4)` runs `process_single` over every matching file in a directory using `multiprocessing.Pool` (each worker is a fresh process — `_process_worker` is a picklable top-level function taking a tuple of args). Optional `meta_csv` supplies per-image artefact metadata (a `filename` column plus arbitrary metadata columns) so batch runs of different collections can carry accurate `type`/`location`/`period` fields into each record rather than leaving them as `"unknown"`.

CLI: `python -m src.pipeline single <image> [--script ...] [--method ...] [--dstretch] [--pdf] [--type ...] [--location ...] [--period ...]` and `python -m src.pipeline batch <dir> [--workers N] [--meta-csv ...]`.

---

## 11. Web Application Architecture

### 11.1 Backend: FastAPI (`api/main.py`)

**Endpoints** (unchanged surface area, but the `ocr` stage is now live rather than a `"not yet implemented"` skip):

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/images` | GET | List all raw images with metadata |
| `/api/images/{id}/thumbnail` | GET | Return 400px max thumbnail |
| `/api/process` | POST | Submit batch processing job (`stages` can now include `"ocr"`) |
| `/api/jobs/{id}` | GET | Poll job status and results |
| `/data/*` | GET | Static file serving for outputs |
| `/test_output/*` | GET | Static file serving for the ad-hoc `test_output/` directory used by tuning scripts (mounted only if the directory exists) |

`ProcessRequest` now also accepts `stage_options: dict[str, dict[str, str]]` — per-stage option overrides (e.g. `{"binarise": {"method": "otsu"}, "ocr": {"script": "tamil"}}`) passed straight through to `api/pipeline.py`'s `run_stage()`.

**Image ID system:** unchanged — `collection__subfolder__filename.ext` (double underscore delimiter), generated by `image_id_for_path()`.

### 11.2 API Pipeline Adapter (`api/pipeline.py`)

`run_stage(image_id, stage, options)` now dispatches four real stages (`preprocess`, `enhance`, `binarise`, `ocr`) instead of Stage 1–3 only; anything else still returns `{"status": "skipped", "reason": "Stage '...' not yet implemented"}` (this is where `translate`/`record` would plug in once built).

- `_run_enhance()` accepts an explicit `mode` option (defaulting to `"superres"` at the API layer — note this differs from `src.enhance.enhance()`'s own `"auto"` default, so the web UI's default request always asks for super-resolution unless the caller overrides it) and returns a cached result immediately if the same-mode output file already exists on disk.
- `_run_binarise()` re-detects document type from the *raw* image to decide binarisation source: palm-leaf images binarise from raw; everything else prefers the latest `*_enhanced_superres*.jpg`, falling back to preprocessed, falling back to raw — deliberately never picking a `dstretch`-mode enhancement (§5.9).
- `_run_ocr()` picks the best available input by preference order — binarised → enhanced → preprocessed → raw — and returns a trimmed summary (`script`, `overall_confidence`, `confidence_status`, a 200-character `text_preview`, `engine_used`) rather than the full transcription, keeping job-status API responses small; the full JSON is still written to `data/transcriptions/` and reachable via `url`.

### 11.3 Frontend: React 19 + TypeScript + Vite

**Stack:** React 19.2, TypeScript 6.0, Vite 8.0, Tailwind CSS v4.2. (TanStack Query is no longer a listed dependency in `web/package.json` — data fetching now goes through the custom `useImages`/`useJob` hooks in `web/src/hooks/` directly.)

**Components** (`web/src/components/`): `ImageGrid.tsx`, `ImageCard.tsx`, `StagePanel.tsx`, `ProgressBar.tsx`, `ResultViewer.tsx`, `ComparisonSlider.tsx`.

`ImageGrid.tsx` is the interactive manuscript browser added for large-collection management:

- Free-text **search** (matching filename, language, or collection) is passed through `useDeferredValue` so typing stays responsive even while filtering thousands of images.
- **Language filter chips** are derived from the loaded image set (`Map<language, count>`), each showing a live count.
- **Incremental rendering**: only `visibleCount` (starting at 300) images render at once, with a "Load more (N remaining)" button appending another 300 — a plain windowing strategy (not virtualization) that keeps the DOM manageable on collections with many thousand raw images.
- **Grouped-by-language sections**, each with its own responsive grid (2 columns on mobile up to 6 on large screens).
- **Selection state** (`selected: Set<string>`) is lifted to the parent via `onSelectionChange`, with "Select all" (applies to the currently *filtered* set, not the whole library) and "Deselect all" actions, and a live selected-count badge.

`types.ts`'s `StageResult` now carries OCR-specific fields (`text`, `text_preview`, `script`, `overall_confidence`, `confidence_status`, `engine_used`) alongside the original `status`/`url`/`error`, and `StageName` includes `"ocr"` and `"translate"` (the latter still unused by the backend). `ResultViewer.tsx` renders these — script name, confidence percentage with status colour-coding, and the extracted text preview — for any stage result that includes them.

---

## 12. Data Organisation

```
data/
├── raw/                — Source images (read-only, never modified)
│   ├── tamil_stone/
│   ├── kannada_stone/
│   ├── malayalam_stone/
│   ├── telugu_stone/
│   └── tulu_stone/
├── preprocessed/       — Stage 1 output (JPEG, quality 95)
├── enhanced/            — Stage 2 output (JPEG, quality 95) — filename now encodes mode, e.g. {stem}_enhanced_superres.jpg / _dstretch.jpg / _mild.jpg
├── binarised/           — Stage 3 output (PNG, lossless, white text on black)
├── thumbnails/          — UI preview images (cached, 400px)
├── transcriptions/      — Stage 4 output (JSON) — IMPLEMENTED
├── records/             — Stage 6 output (JSON, INS-YYYY-NNNN.json) — IMPLEMENTED
└── translations/        — Stage 5 output — not yet produced by anything (Phase 2 pending)

tessdata/                — Bundled Tesseract language data (eng, tam, kan, tel, mal)
models/weights/          — RealESRGAN_x4plus.pth (+ an unused anime variant); unet_binarise.pth / docentr_binarise.pth expected here if trained
outputs/exports/         — PDF exports from src.record.export_pdf()
```

**Naming convention:** `{image_stem}_{stage}.{ext}`, e.g. `image001_preprocessed.jpg`, `image001_enhanced_superres.jpg`, `image001_binarised.png`, `image001_transcription.json`.

**Non-destructive rule:** raw images are never modified; each stage writes to its own directory. Unchanged from the original design.

---

## 13. Testing

Located in `tests/` using pytest. Test coverage now spans all four implemented processing stages plus the API:

- **test_preprocess.py** — EXIF correction, CLAHE brightness shift, white balance neutralisation, border crop dimensions.
- **test_enhance.py** — denoising, Real-ESRGAN 2× upscale, DStretch channel variance change, unsharp mask edge enhancement, auto-mode routing.
- **test_binarise.py** — binary-output validation (only 0/255 present), document-type detection (now including `metal_plate`), noise blob removal, palm-leaf/stone/copper-plate routing.
- **test_ocr.py** — script detection, `ocr_tesseract`/`ocr_easyocr` schema and graceful-degradation behaviour, ensemble merge logic, line grouping, full `transcribe()` behaviour including the Brahmi/Grantha manual-transcription path, output-path building.
- **test_api.py** — REST endpoints, error cases, and now also `ocr` and `enhance` stage execution through `run_stage`.

Outside `tests/`, the repository also carries a number of ad-hoc, non-pytest scripts left over from binarisation tuning work — `single_image.py`, `single_image_test.py`, `palm_leaf.py`, and root-level scripts like `run_ocr_image9.py`, `draw_ocr_boxes_image9.py`, `ocr_test.py`, `test_ocr_reconstruction.py` — used for interactively diagnosing individual problem images (see §5.7) rather than as part of the automated suite.

---

## 14. Technology Stack

### Image Processing

| Library | Version | Purpose |
|---|---|---|
| OpenCV (cv2) | ≥4.9.0 | Morphology, thresholding, denoising, colour space conversion |
| Pillow | ≥10.3.0 | Image I/O, EXIF handling |
| NumPy | ≥1.26.0 | Array operations, linear algebra (DStretch) |
| scikit-image | ≥0.23.2 | Sauvola thresholding, self-reference PSNR/SSIM |

### AI/ML & OCR

| Library | Version | Purpose |
|---|---|---|
| PyTorch | ≥2.0.0 | U-Net and DocEnTr DL binarisation |
| torchvision | ≥0.15.0 | Required by BasicSR/Real-ESRGAN |
| BasicSR | ≥1.4.2 | RRDBNet architecture for Real-ESRGAN |
| Real-ESRGAN | ≥0.3.0 | Pre-trained super-resolution weights |
| pytesseract | 0.3.13 (installed, **not yet pinned in `requirements.txt`**) | Tesseract OCR bindings |
| easyocr | 1.7.2 (installed, **not yet pinned in `requirements.txt`**) | Deep-learning OCR engine |
| fpdf2 | optional | PDF export in `src.record.export_pdf` |

### Web

| Component | Version | Purpose |
|---|---|---|
| FastAPI | ≥0.111.0 | REST API |
| Uvicorn | ≥0.29.0 | ASGI server |
| React | 19.2.4 | Frontend |
| Vite | 8.0.4 | Build tool & dev server |
| Tailwind CSS | 4.2.2 | Styling |
| TypeScript | 6.0.2 | Type safety |

---

## 15. Performance

### Processing Time (CPU, 3000×4000px image, indicative)

| Stage | Duration | Bottleneck |
|---|---|---|
| Preprocessing | 1–2 seconds | CLAHE tile computation |
| Enhancement | 1–3s (`mild`) or 15–25s (`superres`) | Real-ESRGAN tile processing, when used |
| Binarisation | 0.5–2 seconds | Sauvola local thresholding / connected-component filtering |
| OCR (Tesseract + EasyOCR) | 1–5 seconds | EasyOCR model inference (CPU) |
| **Total (typical stone image, `mild` enhancement)** | **~4–10 seconds** | OCR + binarisation |
| **Total (low-res image, `superres` enhancement)** | **~20–30 seconds** | Real-ESRGAN |

### Storage per Image

| Stage | Format | Size (3000×4000px) |
|---|---|---|
| Raw | JPEG | 3–5 MB |
| Preprocessed | JPEG | 2–4 MB |
| Enhanced (`superres`, 2×) | JPEG | 8–15 MB |
| Enhanced (`mild`, same resolution) | JPEG | 2–4 MB |
| Binarised | PNG | 200–500 KB |
| Transcription | JSON | a few KB |
| Record | JSON | a few KB |

GPU support via PyTorch CUDA still provides an estimated 5–8× speedup on Real-ESRGAN when `superres` mode is used. EasyOCR is currently forced to `gpu=False` in `_get_easyocr_reader()` — a straightforward future win would be making that configurable.

---

## 16. What's Actually Left (Revised Phase 2 Scope)

The original Phase 2 plan bundled OCR, translation, and record assembly together. That plan is now mostly done — only translation remains:

### Stage 5 — Translation (Not Started)

No `src/translate.py` exists. The intended design (Helsinki-NLP OPUS-MT for post-10th-century texts, Claude/GPT-4 API fallback for classical/archaic forms) is unchanged from the original plan, and `src/record.py` already has the integration point ready (`assemble_record(..., translation=...)`).

### Brahmi & Grantha OCR — Still No Model

`SCRIPT_CONFIG` has both scripts configured with `tesseract_lang: None, easyocr_lang: None`; `transcribe()` correctly short-circuits to `manual_transcription_required` rather than attempting and failing. Fine-tuning on the Brahmi Character Dataset (arxiv.org/abs/2501.01981) remains unstarted.

### Omeka S Export — Not Started

No code targets the public-portal export format mentioned in the original plan; `export_pdf()` covers researcher-facing PDF output only.

---

## 17. Known Limitations

- **In-memory job store:** `api/jobs.py` job progress is not persistent across server restarts. No PostgreSQL/TinyDB has been added.
- **Grey-world AWB limitation:** fails on scenes dominated by a single colour (e.g., moss-covered stone). No alternative AWB implemented.
- **DL binarisation without weights:** U-Net and DocEnTr fall back to Sauvola whenever weights files are absent (the default state — no weights ship in the repo) or confidence is below 0.65.
- **Real-ESRGAN import failure:** enhancement gracefully falls back to `mild` mode, logging a warning, if BasicSR/Real-ESRGAN is not importable or throws.
- **`pytesseract`/`easyocr` are installed in the dev environment but not yet declared in `requirements.txt`** — a fresh `pip install -r requirements.txt` will not give a working OCR stage; this should be fixed before onboarding a new environment.
- **Per-image binarisation overrides are dimension-matched as a fallback** (§5.7): a coincidentally same-sized new image could be silently routed to a bespoke pipeline tuned for a different photograph. These overrides are a debugging/demo convenience, not a generalisable design, and should be removed or made opt-in before processing a larger unseen dataset.
- **`detect_script()` is a coarse heuristic**, not a trained classifier — it reliably distinguishes Devanagari from "everything else" but defaults every other script to Tamil regardless of actual content; multi-script batches need the caller to pass `script=` explicitly rather than relying on `auto`.
- **EasyOCR is hard-coded to CPU** (`gpu=False`) in `src/ocr.py`.
- **Translation and Omeka S export are unimplemented** (§16).

---

## 18. Possible Examination Questions

### Conceptual Questions

**Q1. What is the fundamental difference between preprocessing and enhancement?**
Preprocessing removes distortions *introduced by the scanning process* (orientation errors, colour casts, uneven exposure, scan borders) without adding information. Enhancement actively improves *legibility of the artefact itself* using AI super-resolution, denoising, and pigment-reveal algorithms that synthesise or recover detail not cleanly captured in the raw scan.

**Q2. Why was the stone binarisation method changed from a black-hat morphological transform to bilateral filtering + adaptive Sauvola thresholding?**
The black-hat approach isolated carved grooves by spatial scale using one fixed-size structuring element, which worked well for a narrow range of image resolutions and stone textures but did not generalise across the dataset's wide variation in resolution, contrast, and lighting. The current `binarise_stone()` instead tunes its Sauvola window size and `k` constant directly from each image's own resolution and contrast statistics (`_adaptive_sauvola_params`), and adds a distinct high-resolution branch, trading some of the original method's elegance for measured robustness across more images.

**Q3. Why does palm-leaf binarisation now segment character-by-character instead of applying one adaptive threshold to the whole leaf?**
Illumination gradients run the length of a photographed palm leaf. Even a locally-adaptive method like Sauvola, applied once across the whole image, must compromise between different lighting conditions at different points along the leaf. By first locating character-sized blobs with a rough mask, then re-running Sauvola *inside each blob's own bounding box*, each local threshold only has to handle the near-uniform illumination of that one small crop — eliminating background bleed that a single whole-image pass leaves behind.

**Q4. Why is DStretch-enhanced output never used as binarisation input?**
DStretch performs a decorrelation stretch across RGB channels, deliberately distorting colour relationships to reveal faded pigment or ink. That same distortion breaks the pixel-intensity assumptions Sauvola/Otsu thresholding depend on (e.g. the grey-world-like assumption that ink is simply "darker" than background). `api/pipeline.py`'s `_run_binarise()` therefore always prefers `*_enhanced_superres*` output, never `*_enhanced_dstretch*`, when selecting a binarisation source.

**Q5. Why were PSNR/SSIM redefined to compare an image against its own smoothed version rather than against the original scan?**
There is no ground-truth "correct" version of a historical inscription to diff against — the raw scan itself is degraded, and "enhancement" is supposed to differ from it. Comparing the enhanced output against a mild bilateral-filtered version of itself instead asks a meaningful question: how much high-frequency processing noise (ringing, over-sharpening artefacts) does the pipeline introduce, independent of what the original degraded scan looked like.

**Q6. What is the OCR ensemble strategy and why merge word boxes from both engines?**
`ocr_ensemble()` runs Tesseract and EasyOCR independently, and takes the *entire* transcription (text + boxes) from whichever engine reports the higher mean confidence as primary. It then folds in word boxes from the other ("secondary") engine only at positions the primary engine didn't already cover — giving denser bounding-box coverage (useful for flagging uncertain regions) without letting a lower-confidence engine override the primary engine's actual text.

**Q7. Why does `enhance()` default to `mode="mild"` for most stone images instead of always running Real-ESRGAN super-resolution?**
Empirically, most stone photographs in the dataset are captured at resolutions where the underlying detail is already present — upscaling with Real-ESRGAN adds synthesized detail that isn't always faithful, and can soften genuine texture. `mode="auto"` now reserves `superres` for genuinely low-resolution captures (shorter side < 500px) where detail truly needs to be synthesized, and uses denoise+sharpen only (`mild`) for everything else — a direct empirical revision of the original "always super-resolve stone" assumption.

**Q8. Why does `detect_document_type()` check the image's file path before analysing pixels?**
The dataset's folder structure (`tamil_stone/`, `malayalam_stone/`, etc.) is a highly reliable ground-truth signal that is cheaper and more accurate than re-deriving document type from pixel statistics every time — and pixel-based heuristics occasionally misclassify edge cases (e.g. reddish sandstone flagged as palm leaf by hue alone). The path check is a fast, reliable shortcut; pixel analysis remains the fallback for images without informative paths (e.g. ad-hoc uploads via the web UI).

**Q9. What happens when OCR is requested for a script with no configured engine (Brahmi, Grantha)?**
`transcribe()` checks `SCRIPT_CONFIG` up front; if both `tesseract_lang` and `easyocr_lang` are `None` for the detected/requested script, it short-circuits immediately, returning `status: "manual_transcription_required"` with empty text and zero confidence, without attempting an OCR call that would only fail.

**Q10. How does the pipeline decide what "sourced" image to binarise from, given four possible intermediates (raw, preprocessed, enhanced, and enhanced-with-dstretch)?**
`api/pipeline.py`'s `_run_binarise()` re-runs `detect_document_type()` on the raw image first. Palm-leaf images always binarise from raw (the character-segmentation method does its own local normalisation and doesn't benefit from the general enhancement chain). Everything else prefers the newest `*_enhanced_superres*.jpg`, falling back to the preprocessed image, falling back to raw — and deliberately never to a `dstretch`-mode enhancement output (Q4).

### Architecture Questions

**Q11. Why does `ProcessRequest` carry a `stage_options` dict instead of one global set of parameters?**
Different stages need different tunable parameters (binarisation `method`, enhancement `mode`, OCR `script`), and different images processed in the same batch job may need different values for the same stage (e.g. one image forced to `otsu` binarisation because Sauvola misfires on it). A per-stage options dict keeps each stage's parameters independent and lets the frontend expose stage-specific controls without the API schema needing to change every time a new tunable is added.

**Q12. Why does `process_single()` catch exceptions per-stage rather than letting the whole pipeline fail on the first error?**
Each stage's `_stage()` wrapper logs a failure and returns `None` rather than propagating the exception, and each subsequent stage falls back to the last-successful output path as its input. This means, e.g., a Real-ESRGAN crash on one unusual image still lets binarisation and OCR run against the preprocessed image rather than losing the whole batch item — the resulting record's `processing_log` records exactly which stage failed and why, for later triage.

**Q13. Why does `process_batch()` use `multiprocessing.Pool` rather than threads or `asyncio`?**
The pipeline's per-image work (OpenCV operations, PyTorch inference, Tesseract subprocess calls) is CPU-bound rather than I/O-bound, so Python's GIL would prevent threads from achieving real parallelism. `multiprocessing.Pool` gives each worker its own interpreter and CPU core; the worker function (`_process_worker`) is deliberately a picklable top-level function taking a plain tuple, since multiprocessing needs to serialise call arguments across the process boundary.

**Q14. How does the image ID system handle images with the same filename in different collections?**
IDs are generated from the relative path within `data/raw/`: `collection__subfolder__filename.ext` using double underscore as delimiter. Two images `data/raw/tamil_stone/img001.jpg` and `data/raw/kannada_stone/img001.jpg` produce distinct IDs `tamil_stone__img001.jpg` and `kannada_stone__img001.jpg`.

**Q15. Why is record assembly (`assemble_record`) designed to accept a `None` translation rather than requiring one?**
Translation (Stage 5) is not yet implemented, but record assembly, storage, and PDF export (Stage 6) needed to be usable end-to-end regardless. `assemble_record()` fills in a `status: "phase_2_pending"` placeholder block when no translation is supplied, so every other part of a record — transcription, quality metrics, citation — is still complete and useful today, and the translation field is a strict additive integration point for whenever `src/translate.py` is built.
