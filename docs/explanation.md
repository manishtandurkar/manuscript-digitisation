# Technical Explanation: Inscription Digitisation Project Implementation

**Date:** May 8, 2026
**Phase:** Phase 1 — OCR & Transcription System (Architecture & Implementation)
**Status:** Three processing stages fully implemented with automated testing and web UI integration

---

## 1. Project Context & Objectives

The Inscription Digitisation Project transforms degraded scanned images of historical South Asian artefacts — stone inscriptions, palm leaf manuscripts, copper plates, paper manuscripts, and cave/rock paintings — into high-quality, machine-readable research records.

The core mission is to extract legible text from unclear source images through a multi-stage image processing pipeline. Phase 1 delivers a robust preprocessing, enhancement, and binarisation foundation. OCR/transcription (Stage 4) and beyond are deferred to Phase 2.

### Pipeline Overview

```
Raw Image (JPG/PNG/TIF/AVIF)
    ↓
[Stage 1: Preprocessing]   — Normalise the scan itself (fix orientation, exposure, colour, borders)
    ↓
[Stage 2: Enhancement]     — Improve legibility (AI super-resolution, noise removal, pigment reveal)
    ↓
[Stage 3: Binarisation]    — Convert to strict black/white for OCR
    ↓
[Stage 4: OCR]             — Extract characters (Phase 2)
    ↓
[Stage 5: Translation]     — Convert to English (Phase 2)
    ↓
[Stage 6: Record Assembly] — Bundle all outputs into structured JSON (Phase 2)
```

---

## 2. The Crucial Distinction: Preprocessing vs Enhancement

This distinction is frequently misunderstood and is the conceptual centrepiece of the pipeline.

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
- Super-resolved strokes have soft edges → Unsharp mask sharpens them.

**Analogy:** Enhancement is like a forensic photograph expert enhancing a blurry image — it goes beyond the raw capture to recover what the raw scan could not faithfully represent.

### Side-by-Side Comparison

| Property | Preprocessing | Enhancement |
|---|---|---|
| Goal | Remove scan artefacts | Improve legibility of the artefact |
| Information change | Removes distortion, no net gain | Synthesises/reveals new detail |
| Algorithms | CLAHE, grey-world AWB, crop | Real-ESRGAN, NLM denoise, DStretch |
| AI involved? | No (classical signal processing) | Yes (deep learning for super-resolution) |
| Changes resolution? | No (may reduce via crop) | Yes (2× upscale via Real-ESRGAN) |
| Speed | 1–2 seconds | 15–25 seconds |
| Order in pipeline | Must be first | Requires preprocessed input |

---

## 3. Stage 1: Preprocessing — Technical Deep Dive

**Source:** [src/preprocess.py](../src/preprocess.py)

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

1. Converts BGR → LAB colour space. LAB separates luminance (L channel) from colour (A = green-red axis, B = blue-yellow axis). This is important because histogram equalisation should only touch brightness, not hue.
2. Applies CLAHE to the L channel only (A and B are untouched — colours do not shift).
3. `tileGridSize=(8,8)` divides the image into an 8×8 grid of tiles, equalising each independently. This makes it *adaptive* — a dark shadow in one corner does not affect equalisation of a bright centre.
4. `clipLimit=2.0` caps the histogram gradient to prevent noise amplification. Without clipping, AHE would drastically over-amplify uniform-noise regions.
5. Merges and converts back to BGR.

**Why not global histogram equalisation?** A stone inscription photographed outdoors has sunlit areas and deeply shadowed crevices in the same frame. Global HE would normalise across both, washing out the detail in shadows or blowing out the highlights. CLAHE adapts locally, preserving both.

### 3.3 Grey-World White Balance

```python
def auto_white_balance(img: np.ndarray) -> np.ndarray:
    img_float = img.astype(np.float32)
    channel_means = img_float.reshape(-1, 3).mean(axis=0)   # mean per B, G, R channel
    overall_mean = float(channel_means.mean())               # mean across all channels
    scale = overall_mean / np.maximum(channel_means, 1e-6)   # per-channel scaling factor
    balanced = img_float * scale.reshape(1, 1, 3)
    return np.clip(balanced, 0, 255).astype(np.uint8)
```

**Grey-world assumption:** If an image contains a random variety of colours, the average colour should be neutral grey. Any deviation from grey is a colour cast from the light source.

1. Computes mean pixel value for each channel independently (B mean, G mean, R mean).
2. Computes overall mean across all three.
3. Scales each channel so its mean equals the overall mean — neutralising the cast.
4. Clips to [0, 255] to prevent overflow.

**Limitation:** The grey-world assumption fails if a scene is dominated by a single colour (e.g., a very green moss-covered stone). In practice, for the mixed surfaces of inscriptions, this heuristic is effective.

### 3.4 Border Cropping

```python
def _crop_borders_with_metadata(img: np.ndarray, threshold: int = 10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = ((gray > threshold) & (gray < 255 - threshold)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    points = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(points)
    # Sanity check: reject crops that would remove >75% of the image
    if w < width * 0.25 or h < height * 0.25:
        return img.copy(), (0, 0, width, height)
    cropped = img[y : y + h, x : x + w]
    return cropped, (x, y, w, h)
```

1. Creates a binary mask: pixels with values in the interior range `(10, 245)` are content; near-black and near-white are border.
2. `MORPH_CLOSE` connects fragmented content regions (fills small gaps in the content mask with a 5×5 kernel).
3. `cv2.findNonZero()` locates all non-zero pixels; `cv2.boundingRect()` computes the tightest bounding box around them.
4. **Sanity check:** If the detected content bounding box is less than 25% of the original width or height, the crop is rejected and the original is returned unchanged. This prevents aggressive over-cropping when borders are misdetected.
5. Returns both the cropped image and the crop coordinates `(x, y, w, h)` for audit logging.

**Output:** JPEG at quality 95 saved to `data/preprocessed/`. JPEG is used instead of TIFF to avoid Windows PIL/libtiff compatibility issues; quality 95 is sufficient for subsequent pipeline stages.

---

## 4. Stage 2: Enhancement — Technical Deep Dive

**Source:** [src/enhance.py](../src/enhance.py)

The enhancement chain executes in a fixed order:

```
denoise()  →  enhance_with_realesrgan() OR dstretch()  →  sharpen()
```

### 4.1 Non-Local Means Denoising

```python
def denoise(img: np.ndarray, strength: int = 10) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)
```

`cv2.fastNlMeansDenoisingColored()` is a colour-aware implementation of Non-Local Means (NLM) denoising.

**How NLM works:**
- For each target pixel, it searches a large window (21×21 pixels, the last parameter) for similar patches (7×7 pixels, the second-to-last parameter).
- Each found patch votes on the target pixel's value, weighted by patch similarity (not just distance).
- This is fundamentally different from Gaussian blur, which only considers proximity — NLM finds structurally similar regions anywhere in the window.

**Parameters:**
- `strength=10` for mild noise (default). `strength=20` for heavily damaged images.
- The 7-pixel search patch size and 21-pixel window are empirically chosen for inscription images.

**Why NLM before super-resolution?** Denoising first prevents Real-ESRGAN from amplifying noise artefacts into the upscaled image.

### 4.2 Real-ESRGAN Super-Resolution

```python
def _build_upsampler(model_path: str):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4)
    return RealESRGANer(scale=4, model_path=model_path, model=model,
                        tile=400, tile_pad=10, pre_pad=0, half=False)

@lru_cache(maxsize=2)
def _get_upsampler(model_path: str):
    return _build_upsampler(model_path)

def enhance_with_realesrgan(img, scale=2, model_path=DEFAULT_MODEL_PATH):
    upsampler = _get_upsampler(str(mp.resolve()))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output_rgb, _ = upsampler.enhance(img_rgb, outscale=scale)
    return cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR).astype(np.uint8)
```

**Architecture:** RRDBNet (Residual-in-Residual Dense Block Network). It is the generator from the Real-ESRGAN GAN architecture, trained to super-resolve real-world degraded photographs (not synthetic downsamples).

**Key implementation details:**

- **4× model, 2× output (`outscale=2`):** The underlying model was trained for 4× upscaling but `outscale=2` instructs it to output only 2×. This is intentional — full 4× over-smooths character strokes on these images.
- **Tiling (`tile=400`, `tile_pad=10`):** Large inscriptions (3000×4000px) would exhaust CPU/GPU memory in a single pass. The image is split into 400×400px tiles with a 10-pixel overlap (padding). Each tile is processed independently and the results are stitched. The 10px overlap prevents visible seam artifacts at tile boundaries.
- **`half=False`:** Uses full float32 precision. `half=True` (float16) is faster on supported GPUs but can introduce precision artifacts for text.
- **Auto-download:** On first run, weights (~63MB) are downloaded from the official GitHub release URL to `models/weights/RealESRGAN_x4plus.pth`.
- **LRU cache:** `@lru_cache(maxsize=2)` keeps up to 2 loaded models in memory. Loading a PyTorch model from disk takes 2–3 seconds — caching avoids this cost on repeated calls during batch processing.

**Why Real-ESRGAN over classical upscaling?** Bicubic and Lanczos interpolation produce blurry edges. Real-ESRGAN, trained on real degraded images, synthesises texture and edges plausibly, making character strokes sharper and more distinct.

### 4.3 DStretch Decorrelation Stretch

```python
def dstretch(img: np.ndarray, colour_space: str = "LAB") -> np.ndarray:
    img_float = img.astype(np.float64) / 255.0
    flat = img_float.reshape(-1, 3)
    mean = flat.mean(axis=0)
    centered = flat - mean
    cov = (centered.T @ centered) / (n - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    stretch_matrix = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues + 1e-10)) @ eigenvectors.T
    stretched = centered @ stretch_matrix
    # Linear rescale to [0, 255]
    lo, hi = stretched.min(axis=0), stretched.max(axis=0)
    stretched = (stretched - lo) / (hi - lo + 1e-10)
    return (stretched.reshape(img_float.shape) * 255).astype(np.uint8)
```

DStretch was originally developed for rock art analysis and is used here for cave paintings and heavily weathered stone.

**Mathematical explanation:**
1. Represents each pixel as a point in 3D RGB colour space.
2. Computes the covariance matrix — describes how correlated the channels are (if R and G tend to vary together, they are correlated).
3. Eigenvalue decomposition identifies the principal colour components (the axes along which colour varies most).
4. The stretch matrix is `V * diag(1/sqrt(λ)) * V^T` where V are eigenvectors and λ are eigenvalues. This has the effect of equalising variance along each principal colour axis, essentially making all colour directions equally prominent.
5. The result is linearly rescaled to [0, 255].

**Why this reveals hidden pigment:** Faded cave paintings may have colour differences between painted and unpainted rock of only 2–3 RGB units per channel — invisible to the human eye and camera noise. DStretch amplifies these tiny differences to fill the full colour range, making faded outlines visible.

**When it's used:** `mode="dstretch"` or `use_dstretch=True` in the `enhance()` call. Cave painting images use this instead of Real-ESRGAN.

### 4.4 Unsharp Mask Sharpening

```python
def sharpen(img: np.ndarray, amount: float = 1.5) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)
```

**Formula:** `output = original × (1 + amount) − blur × amount`

This is equivalent to: `output = original + amount × (original − blur)`

The term `(original − blur)` is a high-pass filter (edge detail). Adding it back to the original amplifies edges without introducing the halo artifacts of naive sharpening.

Applied after Real-ESRGAN because super-resolution softens character boundaries slightly during upscaling.

### Enhancement Modes Summary

| Artefact Type | Pipeline |
|---|---|
| Stone inscriptions | denoise → Real-ESRGAN (2×) → sharpen |
| Cave/rock paintings | denoise → DStretch → sharpen |
| Palm leaf manuscripts | denoise → Real-ESRGAN (2×) → sharpen |

---

## 5. Stage 3: Binarisation — Technical Deep Dive

**Source:** [src/binarise.py](../src/binarise.py)

Binarisation converts a colour or grayscale image to strictly binary (0 or 255). This is a prerequisite for OCR — Tesseract and EasyOCR expect black text on white background.

The binarisation stage is the most complex in Phase 1, implementing **5 distinct methods** plus **automatic document-type routing**.

### 5.1 Document-Type Detection

Before choosing a binarisation path, the stage automatically detects whether the image is a palm leaf manuscript or stone/other inscription:

```python
def detect_document_type(img: np.ndarray) -> str:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_hue = float(hsv[:, :, 0].mean())
    mean_sat = float(hsv[:, :, 1].mean())
    if mean_sat > 40 and 8 <= mean_hue <= 30:
        return "palm_leaf"
    return "stone"
```

- Converts to HSV colour space. Hue (H) runs 0–180 in OpenCV; Saturation (S) runs 0–255.
- Palm leaf manuscripts have a characteristic warm tan/orange background: hue between 8–30 (orange range) and mean saturation above 40 (not achromatic).
- Stone inscriptions have near-grey surfaces: low saturation.
- This detection drives which specialised pipeline runs when `method="sauvola"` is specified.

### 5.2 Method 1: Stone Binarisation (routed via `method="sauvola"` on stone images)

```python
def binarise_stone(img: np.ndarray) -> np.ndarray:
    gray = _to_gray(img)
    # 1. Gaussian smooth: sigma=5 kills <10px grain, carved grooves (15-50px) survive
    smooth = cv2.GaussianBlur(gray, (0, 0), sigmaX=5, sigmaY=5)
    # 2. Black-hat: kernel ~1/12 of shorter edge, elliptical
    k = max(31, min(h, w) // 12)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    black_hat = cv2.morphologyEx(smooth, cv2.MORPH_BLACKHAT, kernel)
    # 3. Normalize; threshold at 75th percentile (top 25% = carved grooves)
    black_hat = cv2.normalize(black_hat, None, 0, 255, cv2.NORM_MINMAX)
    thresh_val = max(int(np.percentile(black_hat, 75)), 30)
    _, binary = cv2.threshold(black_hat, thresh_val, 255, cv2.THRESH_BINARY)
    # 4. Morphological cleanup
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return binary
```

**Key insight — Black-hat morphological transform:**

- The structuring element is an ellipse sized to span a full character stroke width (approx 1/12 of the shorter image edge, minimum 31px).
- Black-hat = `morphological_close(img) − img`. This isolates dark structures *smaller than the kernel* — exactly the carved grooves of inscriptions.
- The stone surface background (larger than the kernel) is suppressed; the carved text (narrower than the kernel) survives as bright peaks in the black-hat image.
- After normalisation, the 75th percentile threshold keeps only the top 25% of response — the carved grooves dominate this peak; stone grain texture sits in the lower tail.
- Floor of 30 prevents thresholding below meaningful signal when response is very low.

**Why not Sauvola directly on stone?** Stone grain has spatial frequency close to character strokes. Sauvola's local window treats both as foreground, producing noisy output. The black-hat transform explicitly separates by scale.

### 5.3 Method 2: Palm Leaf Binarisation (routed via `method="sauvola"` on palm leaf images)

```python
def binarise_palm_leaf(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Adaptive Gaussian thresholding with wide window (31px) and small constant (5)
    binary = cv2.adaptiveThreshold(gray, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 5)
    binary = cv2.bitwise_not(binary)  # invert: ink becomes white
    # Flood-fill from corners to remove background false positives
    mask = np.zeros((h + 2, w + 2), np.uint8)
    for corner in [(0,0), (0,w-1), (h-1,0), (h-1,w-1)]:
        cv2.floodFill(binary, mask, (corner[1], corner[0]), 0)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8))
    return binary
```

**Why a separate path?**
Palm leaf manuscripts have warm tan fibre texture with similar local contrast to ink strokes. The standard Sauvola + CLAHE path (designed for near-achromatic stone) aggressively boosts this fibre texture, producing noise as bad as the signal.

The palm leaf path uses:
- **`ADAPTIVE_THRESH_GAUSSIAN_C`** with a wider 31-pixel window (vs 15 for stone) to average over more fibre texture, reducing false positives.
- **Corner flood-fill:** After inversion, background areas connected to image corners are removed. This exploits the fact that palm leaf text is always enclosed content, never touching the image border.
- **2×2 morphological close:** Smaller than stone's 5×5 because ink strokes on palm leaf are finer and 5×5 would merge adjacent characters.

**Noise removal differs by document type:**
```python
if doc_type == "palm_leaf" and method == "sauvola":
    binary = remove_noise_blobs(binary, min_size=8, min_length=15)
elif doc_type == "stone":
    binary = remove_noise_blobs(binary, min_size=200, min_length=30)
```
Palm leaf uses much smaller thresholds because ink strokes are finer. Stone uses larger thresholds because grain speckles that survive are typically larger than ink stroke fragments.

### 5.4 Method 3: Otsu Global Thresholding

```python
def binarise_otsu(img: np.ndarray) -> np.ndarray:
    gray = _clahe(_to_gray(img))
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
```

**Otsu's method** finds the single global threshold that minimises intra-class variance (equivalently, maximises inter-class variance) between foreground and background. It assumes the image histogram is bimodal — two distinct peaks for background and foreground.

- Applied after CLAHE to improve the bimodal distribution.
- ~50ms on 3000×4000px images — extremely fast.
- Best for clean paper manuscripts with uniform lighting.
- Fails on uneven backgrounds where local contrast varies: a single threshold cannot separate text from the varying background intensity.

### 5.5 Method 4: Adaptive Mean Thresholding

```python
def binarise_adaptive(img: np.ndarray) -> np.ndarray:
    gray = _clahe(_to_gray(img))
    binary = cv2.adaptiveThreshold(gray, 255,
                                    cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 15, 8)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
```

- For each pixel, threshold = mean of 15×15 neighbourhood − constant 8.
- More robust than Otsu on uneven images, simpler than Sauvola.
- Used as a fallback when Sauvola results are suboptimal.

### 5.6 Method 5: Lightweight U-Net (Deep Learning)

```python
class _LightUNet(nn.Module):
    _CH = [1, 32, 64, 128, 256]
    # Encoder: 3 × DoubleConv + MaxPool blocks
    # Bottleneck: DoubleConv at max depth
    # Decoder: 3 × ConvTranspose2d upsampling + skip connections + DoubleConv
    # Head: 1×1 conv → sigmoid
```

- Standard U-Net architecture adapted for single-channel (grayscale) input and binary output.
- **DoubleConv blocks:** Each block is Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → ReLU.
- **Skip connections:** Encoder feature maps are concatenated (not added) to decoder feature maps at matching scales. This preserves spatial detail lost during downsampling.
- **Sigmoid output:** Final layer produces a probability map. Each pixel is P(foreground).
- **Confidence check:** `_binary_entropy_confidence()` computes mean certainty: `1 − mean(pixel entropy)`. If below `_CONFIDENCE_THRESHOLD = 0.65`, falls back to Sauvola automatically.
- Expected weights file: `models/weights/unet_binarise.pth` (trained on THPLMD document dataset).

### 5.7 Method 6: DocEnTr (Patch-ViT Deep Learning)

```python
class _DocEnTr(nn.Module):
    # Patch embedding: unfold image into 8×8 patches → linear projection to 256 dims
    # Transformer encoder: 4 layers, 8 attention heads, d_model=256, ffn_dim=512
    # Decoder: linear projection → reshape to spatial → 3-layer CNN refinement → sigmoid
```

Based on the DocEnTr paper (El-Hajj & Barakat, ArXiv 2209.09921). Uses a Vision Transformer (ViT) encoder.

**How it differs from U-Net:**
- Instead of convolutional feature extraction, images are split into 8×8 patches and treated as a sequence.
- Each patch is projected to a 256-dimensional token embedding.
- Multi-head self-attention enables each patch to attend to every other patch — capturing global document structure that convolutional kernels miss (e.g., understanding that a region is a margin vs text area based on full-document context).
- Output is decoded back from patch tokens to spatial image via linear projection and CNN refinement.

**Padding:** `_pad_to_multiple()` pads images to the nearest multiple of 8 (the patch size) using reflect padding, then crops the output to original size.

**Same confidence fallback** as U-Net: falls back to Sauvola if confidence < 0.65.

### 5.8 CLAHE Pre-Processing Inside Binarisation

All classical methods call `_clahe()` before thresholding:

```python
def _clahe(gray: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
```

This is the same algorithm as Stage 1's `normalise_brightness()` but applied to a grayscale image just before thresholding. Even after Stage 1's colour CLAHE, the grayscale conversion can reveal residual uneven illumination. Applying CLAHE again on the grayscale improves bimodality for thresholding.

### 5.9 Noise Removal

```python
def remove_noise_blobs(binary, min_size=50, min_length=30):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area >= min_size or max(w, h) >= min_length:
            cleaned[labels == label] = 255
    return cleaned
```

- `cv2.connectedComponentsWithStats()` labels every connected foreground region and records its area, bounding box width, and height.
- A component is kept if its area is ≥ `min_size` **OR** its longest dimension is ≥ `min_length`.
- The dual criterion (area OR length) preserves thin ink strokes on palm leaves that have small area but significant length — pure area filtering would delete them.

### 5.10 Binarisation Method Selection Guide

| Document Type | Recommended Method | Reason |
|---|---|---|
| Stone inscription | `sauvola` (auto-routes to `binarise_stone`) | Black-hat separates grain from grooves by scale |
| Palm leaf manuscript | `sauvola` (auto-routes to `binarise_palm_leaf`) | Wide window + corner fill handles warm fibre texture |
| Clean paper manuscript | `otsu` | Fast, accurate on uniform backgrounds |
| Mixed/uncertain | `adaptive` | Safe fallback |
| DL weights available | `unet` or `docentr` | Highest accuracy if trained weights exist |

**Output:** 8-bit single-channel PNG (lossless). 0 = background (white), 255 = foreground/text (black). Standard format recognised by Tesseract OCR.

---

## 6. Web Application Architecture

### 6.1 Backend: FastAPI (`api/main.py`)

**Framework:** FastAPI with Pydantic validation and automatic OpenAPI docs.

**Endpoints:**

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/images` | GET | List all raw images with metadata |
| `/api/images/{id}/thumbnail` | GET | Return 400px max thumbnail |
| `/api/process` | POST | Submit batch processing job |
| `/api/jobs/{id}` | GET | Poll job status and results |
| `/data/*` | GET | Static file serving for outputs |

**Image ID system:** IDs are derived from path: `collection__subfolder__filename.ext` (double underscore delimiter). This creates stable, filesystem-safe identifiers that encode collection hierarchy.

**Job model:** In-memory job store (`api/jobs.py`) with thread-safe operations. Accepts list of image IDs + stages to run. Returns job ID immediately; client polls for completion.

### 6.2 Frontend: React 19 + TypeScript + Vite

**Stack:** React 19, TypeScript 6.0, Vite 8.0, TanStack Query v5, Tailwind CSS v4.

**Data flow:**
1. `useImages()` hook fetches image list from `/api/images` on mount.
2. User selects images + stages → POST to `/api/process` → receive job ID.
3. `useJob()` hook polls `/api/jobs/{id}` at 1–2 second intervals.
4. UI updates in real-time as stages complete.
5. `ResultViewer` displays processed outputs.

**Type safety:** `types.ts` defines `ImageMeta`, `Job`, `StageResult`, `StageName`. API client enforces return types matching backend Pydantic models — prevents runtime mismatches.

### 6.3 Pipeline Adapter (`api/pipeline.py`)

- `list_raw_images()`: Recursively discovers all images in `data/raw/`. LRU-cached.
- `make_thumbnail()`: On-demand 400px thumbnail, JPEG quality 75. Cached to avoid regeneration.
- `run_stage()`: Routes to `_run_preprocess()`, `_run_enhance()`, or `_run_binarise()` based on stage name. Returns `{status, output_paths, error}`.

---

## 7. Data Organisation

```
data/
├── raw/                — Source images (read-only, never modified)
│   ├── tamil_stone/
│   ├── kannada_stone/
│   ├── malayalam_stone/
│   ├── telugu_stone/
│   └── tulu_stone/
├── preprocessed/       — Stage 1 output (JPEG, quality 95)
├── enhanced/           — Stage 2 output (JPEG, quality 95)
├── binarised/          — Stage 3 output (PNG, lossless)
├── thumbnails/         — UI preview images (cached, 400px)
├── transcriptions/     — Stage 4 output (JSON) — Phase 2
├── translations/       — Stage 5 output (JSON) — Phase 2
└── records/            — Final assembled records — Phase 2
```

**Naming convention:** `{image_stem}_{stage}.{ext}` — e.g., `image001_preprocessed.jpg`, `image001_binarised.png`.

**Non-destructive rule:** Raw images are never modified. Each stage writes to a separate directory.

---

## 8. Testing

Located in `tests/` using pytest.

- **test_preprocess.py:** Validates EXIF correction, CLAHE brightness shift, white balance neutralisation, border crop dimensions.
- **test_enhance.py:** Tests denoising noise reduction, Real-ESRGAN 2× upscale, DStretch channel variance change, unsharp mask edge enhancement.
- **test_binarise.py:** Validates all 5 binarisation methods produce binary output (only values 0 and 255), document-type detection HSV logic, noise blob removal, palm-leaf and stone routing.
- **test_api.py:** REST API endpoints, error cases (missing image, bad stage name).

---

## 9. Technology Stack

### Image Processing

| Library | Version | Purpose |
|---|---|---|
| OpenCV (cv2) | 4.9.0 | Morphology, thresholding, denoising, colour space conversion |
| Pillow | 10.3.0 | Image I/O, EXIF handling |
| NumPy | 1.26.4 | Array operations, linear algebra (DStretch) |
| scikit-image | 0.23.2 | Sauvola thresholding algorithm |

### AI/ML

| Library | Version | Purpose |
|---|---|---|
| PyTorch | ≥2.0.0 | U-Net and DocEnTr DL binarisation |
| BasicSR | 1.4.2 | RRDBNet architecture for Real-ESRGAN |
| Real-ESRGAN | 0.3.0 | Pre-trained super-resolution weights |

### Web

| Component | Version | Purpose |
|---|---|---|
| FastAPI | 0.111.0 | REST API |
| Uvicorn | 0.29.0 | ASGI server |
| React | 19.2.4 | Frontend |
| Vite | 8.0.4 | Build tool & dev server |
| TanStack Query | 5.99.1 | Server state management |
| Tailwind CSS | 4.2.2 | Styling |
| TypeScript | 6.0.2 | Type safety |

---

## 10. Performance

### Processing Time (CPU, 3000×4000px image)

| Stage | Duration | Bottleneck |
|---|---|---|
| Preprocessing | 1–2 seconds | CLAHE tile computation |
| Enhancement (denoise + Real-ESRGAN) | 15–25 seconds | Real-ESRGAN tile processing |
| Binarisation (stone path) | 0.5–1 second | Black-hat morphological transform |
| **Total** | **16–28 seconds** | Super-resolution |

### Storage per Image

| Stage | Format | Size (3000×4000px) |
|---|---|---|
| Raw | JPEG | 3–5 MB |
| Preprocessed | JPEG | 2–4 MB |
| Enhanced (2×) | JPEG | 8–15 MB |
| Binarised | PNG | 200–500 KB |

GPU support via PyTorch CUDA provides estimated 5–8× speedup on Real-ESRGAN. Tiling (400px tiles) prevents memory exhaustion regardless of image size.

---

## 11. Phase 2 Scope

### Stage 4 — OCR & Transcription (Designed, Not Yet Implemented)

**Dual-engine ensemble:**
- **Tesseract OCR:** CPU-based, mature Indic language support (Tamil, Sanskrit, Kannada, Telugu, Malayalam, Devanagari).
- **EasyOCR:** Deep learning engine, better on rotated/degraded text, occasionally over-confident.
- Ensemble: per-word confidence weighting, selecting best candidate from each engine.

**Confidence tiers:**
- ≥ 0.85: Verified (auto-accept)
- 0.60–0.84: Review needed (flag for human check)
- < 0.60: Uncertain (flag for manual transcription)

**Brahmi & Grantha scripts:** No off-the-shelf models available. Phase 2 will fine-tune on the Brahmi Character Dataset (arxiv.org/abs/2501.01981).

### Stage 5 — Translation
Helsinki-NLP OPUS-MT models for post-10th century texts. Claude/GPT-4 API fallback for classical/archaic language forms.

### Stage 6 — Record Assembly
Structured JSON output with PDF export and citation formatting. Export to Omeka S standard format for public portal.

---

## 12. Known Limitations

- **In-memory job store:** Job progress is not persistent across server restarts. Phase 2 will add PostgreSQL or TinyDB.
- **Grey-world AWB limitation:** Fails on scenes dominated by a single colour (e.g., moss-covered stone). No alternative AWB is currently implemented.
- **DL binarisation without weights:** U-Net and DocEnTr automatically fall back to Sauvola when weights files are absent or model confidence is below 0.65.
- **Real-ESRGAN import failure:** If BasicSR/Real-ESRGAN is not installed, the enhancement stage gracefully falls back by logging a warning and skipping super-resolution rather than crashing.

---

## 13. Possible Examination Questions

### Conceptual Questions

**Q1. What is the fundamental difference between preprocessing and enhancement?**
Preprocessing removes distortions *introduced by the scanning process* (orientation errors, colour casts, uneven exposure, scan borders) without adding information. Enhancement actively improves *legibility of the artefact itself* using AI super-resolution, denoising, and pigment reveal algorithms that synthesise or recover detail not cleanly captured in the raw scan.

**Q2. Why is CLAHE applied twice — once in preprocessing and once inside binarisation?**
Stage 1's CLAHE operates on the LAB L-channel of the colour image to normalise brightness while preserving colour. Stage 3's CLAHE operates on the grayscale conversion just before thresholding. Even after Stage 1, the grayscale collapse of colour information can produce a histogram that is less bimodal than optimal for thresholding. The second CLAHE improves the separation of foreground/background peaks.

**Q3. Why does the Real-ESRGAN integration use a 4× model but output only 2×?**
The 4× model was trained to produce 4× resolution but the pipeline sets `outscale=2`. At 4× output, the model over-smooths character strokes, softening edges that need to be sharp for OCR. Empirical testing found 2× output gives the best legibility-to-artifact ratio for historical inscription images.

**Q4. Explain the Black-hat morphological transform and why it's used for stone inscriptions.**
Black-hat = `morphological_close(image) − image`. It isolates dark features smaller than the structuring element. For stone inscriptions, carved grooves are the smallest dark features (narrower than the structuring element). Stone grain texture, which is also dark, is at a similar spatial scale but responds less strongly. After normalisation, thresholding at the 75th percentile retains only the strongest responses (the deepest carved areas), suppressing grain texture that sits in the lower tail of the response distribution.

**Q5. Why are palm leaf manuscripts and stone inscriptions handled by completely different binarisation pipelines?**
Palm leaf manuscripts have warm, coloured fibre texture that has similar local contrast to ink strokes. The stone pipeline's CLAHE + Sauvola would treat both as foreground, producing noisy output. The palm leaf pipeline uses a wider adaptive window to average over fibre texture, then exploits the topological property that ink strokes are always interior content (not connected to image borders) to remove background regions via flood fill from corners.

**Q6. What is the grey-world white balance assumption and when does it fail?**
Assumes the average colour across a natural image should be neutral grey. Any per-channel deviation from the overall mean indicates a colour cast from the light source. Fails when the scene is dominated by a single colour (e.g., moss-covered stone with large green regions), because the grey-world assumption is violated and the correction would introduce an opposite colour cast.

**Q7. Why is DStretch used for cave paintings but not stone inscriptions?**
DStretch amplifies colour differences between channels, making invisible pigment visible. Cave paintings have faded colour pigments — the signal is chromatic (colour differences between rock and paint). Stone inscriptions have monochromatic structure — the signal is luminance (carved depth creating shadows). DStretch provides no benefit for luminance-based structure; it may even obscure it by distorting colour channels.

**Q8. What is Non-Local Means denoising and how does it differ from Gaussian blur?**
Gaussian blur weights pixels by spatial distance — nearby pixels contribute more. NLM weights pixels by patch similarity regardless of distance. For each target pixel, it searches a large window for similar 7×7 patches across the image, and the target pixel's new value is a weighted mean of all similar patch centres. This preserves edges and text stroke boundaries because the algorithm finds similar patches on the same stroke elsewhere in the image and uses them to denoise, rather than blurring across edge boundaries.

**Q9. Why does the `remove_noise_blobs` function use an OR condition (area >= min_size OR length >= min_length)?**
A pure area threshold would delete thin, elongated strokes — common on palm leaf manuscripts where ink is applied in narrow lines with small total area but significant length. The OR condition preserves these strokes by recognising that a long thin component is more likely to be a character stroke than a dust speck, even if its area is small.

**Q10. How does the pipeline prevent one stage from corrupting the original image?**
Raw images in `data/raw/` are opened read-only (only `cv2.imread()` / `PilImage.open()` are called; no write operations touch this directory). Each stage writes its output to a separate directory (`data/preprocessed/`, `data/enhanced/`, `data/binarised/`). The directory structure and output naming convention (`{stem}_{stage}.ext`) enforce this separation at the filesystem level.

### Architecture Questions

**Q11. Why was React + FastAPI chosen over the original Gradio plan?**
Gradio offers rapid prototyping but limited UI control. React + FastAPI enables rich features like before/after comparison sliders, real-time job polling via TanStack Query, and a type-safe API contract enforced by TypeScript + Pydantic. The additional upfront complexity is justified by the interactive nature of the research workflow.

**Q12. How does the tiling mechanism in Real-ESRGAN prevent tile boundary artifacts?**
Tiles are processed with a 10-pixel overlap (`tile_pad=10`). After upscaling, the RealESRGANer stitches tiles by discarding the padded overlap region and using only the interior of each tile for final output. Because each tile was processed with neighbour context (the 10px border), the interior region near the tile boundary is computed with accurate context and stitches seamlessly.

**Q13. What happens if the U-Net or DocEnTr model is unavailable or low-confidence?**
Both DL methods call `_dl_infer()`, which returns `(prob_map, confidence)`. If `model is None` (weights file absent or torch not installed) or `confidence < _CONFIDENCE_THRESHOLD` (0.65), the function returns `(None, 0.0)`. The calling function then falls back to Sauvola. This graceful degradation ensures the pipeline always produces output even without trained weights.

**Q14. How does the image ID system handle images with the same filename in different collections?**
IDs are generated from the relative path within `data/raw/`: `collection__subfolder__filename.ext` using double underscore as delimiter. Two images `data/raw/tamil_stone/img001.jpg` and `data/raw/kannada_stone/img001.jpg` produce distinct IDs `tamil_stone__img001.jpg` and `kannada_stone__img001.jpg`. Special characters in path components are sanitised for filesystem safety.
