# Presentation Reference & Study Guide: Inscription & Manuscript Digitisation

This document serves as your single-source reference to prepare for tomorrow's final interdisciplinary project presentation (evaluated for 100 marks). It covers the folder structure, technical implementations, mathematical formulas, research challenges, and expected examiner questions.

---

## 1. High-Level Folder Walkthrough

Below is the repository structure and the exact, simple purpose of each directory:

| Folder Path | Simple Purpose | Implementation Details |
|---|---|---|
| [`src/`](file:///c:/6th%20semester%20EL's/Interdisciplinary%20project/Implementation/manuscript-digitisation/src/) | Core Algorithms | Python functions executing image preprocessing, filtering, binarisation, metrics, and OCR. |
| [`api/`](file:///c:/6th%20semester%20EL's/Interdisciplinary%20project/Implementation/manuscript-digitisation/api/) | Backend Web Server | FastAPI application exposing endpoints (`/api/images`, `/api/process`, `/api/jobs`) to query database and start pipeline runs. |
| [`web/`](file:///c:/6th%20semester%20EL's/Interdisciplinary%20project/Implementation/manuscript-digitisation/web/) | Frontend Web Interface | React 19 + Vite app enabling users to view galleries, monitor pipelines, and compare images side-by-side. |
| [`data/`](file:///c:/6th%20semester%20EL's/Interdisciplinary%20project/Implementation/manuscript-digitisation/data/) | File Storage | Non-destructive directories (`raw/`, `preprocessed/`, `enhanced/`, `binarised/`, `transcriptions/`, `records/`). |
| [`models/`](file:///c:/6th%20semester%20EL's/Interdisciplinary%20project/Implementation/manuscript-digitisation/models/) | Deep Learning Weights | Stores local PyTorch weights (`RealESRGAN_x4plus.pth`) for neural network inferences. |
| [`scripts/`](file:///c:/6th%20semester%20EL's/Interdisciplinary%20project/Implementation/manuscript-digitisation/scripts/) | Automation & Evaluation | Scripts to automate batch runs (`process_representative_samples.py`) and compute ECE metrics (`evaluate_all.py`). |
| [`scratch/`](file:///c:/6th%20semester%20EL's/Interdisciplinary%20project/Implementation/manuscript-digitisation/scratch/) | Research Sandbox | Where we prototyped algorithms, ran grid searches on parameters, and benchmarked filters. |
| [`tests/`](file:///c:/6th%20semester%20EL's/Interdisciplinary%20project/Implementation/manuscript-digitisation/tests/) | Unit Verification | Pytest scripts verifying step outputs, image dimensions, and API endpoints. |

---

## 2. Stage-by-Stage Implementation & Mathematics

The pipeline consists of 4 main stages delivered in Phase 1:

### Stage 1: Preprocessing ("Fix the Scan")
*   **Purpose:** Removes distortions introduced by cameras/scanners (rotation, uneven lighting, color casts, borders).
*   **Algorithms & Math:**
    *   **EXIF Correction:** `ImageOps.exif_transpose` reads camera rotation metadata and physically rotates the pixel matrix upright.
    *   **CLAHE (Contrast Limited Adaptive Histogram Equalisation):** Converts BGR to LAB color space. Applies local histogram equalisation on the L (Luminance) channel in an $8\times8$ grid:
        $$\text{Clip Limit} = 2.0$$
        This balances shadows (like book spine folds) without altering the hues of the document.
    *   **Grey-World Auto White Balance:** Assumes the average color of a scene is neutral grey. Computes scaling factors ($S_C$) per color channel $C \in \{B, G, R\}$:
        $$S_C = \frac{\text{Overall Mean}}{\text{Mean}_C}$$
        Multiplies the pixel values by $S_C$ to neutralize warm or cool lighting casts.
    *   **Border Cropping:** Applies local thresholding to find non-zero content coordinates and draws the tightest bounding box:
        $$[\text{xmin}, \text{ymin}, \text{width}, \text{height}] = \text{cv2.boundingRect}(\text{points})$$
        If width/height are $<25\%$ of the original size, the crop is skipped to prevent over-cropping.

---

### Stage 2: Enhancement ("Recover the Text Details")
*   **Purpose:** Actively reconstructs character edges, removes noise, and reveals faded pigments.
*   **Algorithms & Math:**
    *   **Non-Local Means Denoising:** Compares $7\times7$ pixel patches inside a $21\times21$ search window. Pixels are averaged based on patch similarity rather than physical distance. This eliminates sensor grain while preserving the sharp boundaries of letter strokes.
    *   **AI Super-Resolution (Real-ESRGAN):** Uses a deep neural network (RRDBNet - Residual-in-Residual Dense Blocks) to predict and synthesize sharp outlines. We run a $4\times$ model but restrict output to $2\times$ (`outscale=2`) to prevent the network from hallucinating fake strokes.
    *   **DStretch (Decorrelation Stretch):** For faded pigments. Projects RGB vectors onto their principal eigenvectors, stretches their variance along these components to remove correlation, and projects them back:
        $$\text{Stretched} = (I - \mu) \cdot (V \cdot \Lambda^{-1/2} \cdot V^T)$$
        This amplifies color differences that are invisible to the human eye.
    *   **Unsharp Masking:** Isolates high-frequency edges by subtracting a Gaussian blurred image from the original, scaling them, and adding them back:
        $$\text{Output} = \text{Original} + \text{Amount} \cdot (\text{Original} - \text{Blur})$$

---

### Stage 3: Binarisation ("Segment Text from Background")
*   **Purpose:** Converts color/grayscale images to pure black-and-white (binary) masks.
*   **Algorithms & Math:**
    *   **Document-Type Routing:** Analyzes the average HSV values. If $8 \le \text{Hue} \le 30$ and Saturation $> 40$, it classifies the file as a `"palm_leaf"` and runs the palm leaf pipeline; otherwise, it runs the `"stone"` pipeline.
    *   **Stone Binarisation:** Runs a morphological Black-Hat filter to isolate narrow carved strokes, normalizes contrast, and thresholds at the 75th percentile of brightness values.
    *   **Palm Leaf Binarisation:** Normalizes L-channel brightness via CLAHE, runs Sauvola thresholding, and intersects it via a bitwise `AND` with an Otsu mask computed on the color A-channel. This isolates color-neutral ink from warm orange background fibers.
    *   **Connected Component Noise Cleanup:** Identifies connected pixel components. It preserves components only if they satisfy:
        $$\text{Area} \ge \text{min\_size} \quad \text{OR} \quad \max(\text{width}, \text{height}) \ge \text{min\_length}$$
        This prevents thin, elongated strokes (common in circular scripts) from being erased.

---

### Stage 4: Quality Metrics & Evaluation (ECE Validation)
*   **Purpose:** Quantitatively validates pipeline quality without requiring ground-truth images.
*   **Algorithms & Math:**
    *   **Self-Reference PSNR/SSIM:** Compares the enhanced image against a pseudo-reference generated by applying an edge-preserving bilateral filter to the enhanced image itself.
        $$\text{PSNR} = 10 \cdot \log_{10} \left( \frac{255^2}{\text{MSE}} \right) \ge 30.0\text{ dB}$$
        $$\text{SSIM} = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)} \ge 0.85$$
    *   **Contrast-to-Noise Ratio (CNR):** Measures text legibility against background noise:
        $$\text{CNR} = \frac{|\mu_{\text{fg}} - \mu_{\text{bg}}|}{\sqrt{\sigma_{\text{fg}}^2 + \sigma_{\text{bg}}^2}} \ge 1.5$$
    *   **Ink Coverage Check:** Guarantees that the percentage of white text pixels in the final binary image is within normal writing limits:
        $$0.5\% \le \text{Ink Coverage} \le 45.0\%$$

---

## 3. Core Research & Engineering Challenges

To present this as a **research-driven project**, highlight these 4 key engineering challenges that you faced and resolved:

### Challenge 1: Stone Texture vs. Carved Grooves (Scale Overlap)
*   **The Problem:** Granite stone has a rough texture with gaps and pits. These pits have the same local contrast and colors as the carved letters. Standard adaptive thresholding (Sauvola) processes these pits as text, creating a noisy image that causes OCR to fail.
*   **Our Solution:** We separated features based on **geometric scale** rather than brightness. We applied a morphological Black-Hat filter using an elliptical structuring element sized to match the chisel width:
    $$\text{Kernel Size } (k) = \min(\text{height}, \text{width}) // 15$$
    Since the stone background texture is larger than the kernel, it is erased, leaving only the narrow carved letters.

### Challenge 2: Palm Leaf Fiber Texture (Color Intersections)
*   **The Problem:** Palm leaves contain vertical/horizontal fibers. When we try to boost contrast to extract faded ink, these fibers stand out, creating horizontal noise lines that blend into the characters.
*   **Our Solution:** We used **Luminance-Chromaticity Isolation**. The leaf background is warm-colored (high values in the green-to-red LAB A-channel), whereas the carbon ink is color-neutral. We run Sauvola thresholding on the L (brightness) channel to find anything dark, run Otsu thresholding on the A (color) channel to find the leaf backing, and perform a bitwise `AND` intersection. This removes fibers because they do not match the ink's color profile.

### Challenge 3: Out-of-Distribution DL Generalisation (Confidence Fallback)
*   **The Problem:** Deep Learning models (like U-Net and DocEnTr) are highly accurate but fragile. When presented with a damaged manuscript of a style they weren't trained on, they generate garbage or high-uncertainty outputs.
*   **Our Solution:** We created a **graceful degradation fallback**. The system calculates the average Shannon Entropy ($H$) of the network's sigmoid probability map:
    $$H = - \frac{1}{N} \sum [p \log_2(p) + (1-p) \log_2(1-p)]$$
    $$\text{Confidence} = 1 - H$$
    If this confidence drops below $0.65$, the system automatically switches to the classical Sauvola local thresholding algorithm.

### Challenge 4: Memory footprint of high-res images (Sliding Tiles)
*   **The Problem:** Archival scans are often $4000\times3000\text{px}$ or larger. Running PyTorch CNN models on these files causes CPU/GPU memory exhaustion (OOM crashes).
*   **Our Solution:** We process the image using a **sliding window tiling mechanism** ($400\times400\text{px}$ tiles) with a $10\text{px}$ overlapping padding. Stitched outputs discard the padding, preventing visible seams.

---

## 4. Expected Examiner Q&A (Prepare for the Viva)

### ECE & Signal Processing Questions

#### **Q1: Why convert BGR to LAB color space before running CLAHE?**
> **Answer:** BGR mixes color and brightness across all three channels. Applying contrast enhancement directly on BGR shifts the hues, changing the colors of the image. LAB separates luminance (L) from color (A/B). Applying CLAHE on the L channel corrects exposure while keeping colors identical to the original.

#### **Q2: Explain the physical intuition behind a Gabor filter.**
> **Answer:** A Gabor filter acts like the visual cortex of the human eye. It is a sinusoidal wave (a stripe pattern) multiplied by a Gaussian envelope (a focus window). By tuning the wave's spacing (frequency) and angle (rotation), it acts as a directional filter that matches parallel text lines while ignoring random background textures.

#### **Q3: How does the FFT noise filter isolate horizontal lines?**
> **Answer:** Repeating horizontal lines represent a periodic signal in the spatial domain. When we compute the 2D Fast Fourier Transform, this periodic signal is concentrated into high-amplitude spikes along the vertical axis of the frequency spectrum. We set these specific spikes to 0 and run the Inverse FFT, removing the lines.

#### **Q4: Why does DStretch work for cave paintings but not stone inscriptions?**
> **Answer:** DStretch is a color decorrelation stretch. It projects color vectors onto their principal components to amplify differences in hue. Cave paintings have color differences (faded pigment on rock). Stone inscriptions are monochromatic—their features are defined by shape and shadow (luminance). DStretch does not help with luminance details and can distort them.

#### **Q5: What is the benefit of Non-Local Means (NLM) denoising over Gaussian blur?**
> **Answer:** Gaussian blur averages neighboring pixels, which smooths out noise but also blurs the sharp edges of characters. NLM searches a larger window for similar patches and averages those. This preserves the boundaries of characters while clearing background noise.

#### **Q6: Why does your CNR metric divide by the standard deviation of the background?**
> **Answer:** Standard deviation measures noise (texture variance). If the stone background is highly textured, a high difference in mean brightness between text and background is still hard to read. Dividing by the standard deviation of the background normalizes the contrast against the background noise.

#### **Q7: What is the math behind Unsharp Masking?**
> **Answer:** Unsharp masking isolates high-frequency details (edges) by subtracting a blurred version of the image from the original. It then adds this detail back to the original image to sharpen outlines:
> $$\text{Output} = I_{\text{orig}} + \text{Amount} \cdot (I_{\text{orig}} - I_{\text{blur}})$$

#### **Q8: What does a negative skewness in `analysis.py` tell you about an image?**
> **Answer:** Negative skewness indicates that the tail of the pixel intensity distribution extends toward the left (dark values). This means the image has a bright background with a few dark features (like black ink on white paper). Positive skewness means a dark image with a few bright highlights.

#### **Q9: Why does `remove_noise_blobs` use an OR condition for area and length?**
> **Answer:** If we only checked area, thin, elongated lines (common in circular scripts) would be deleted because their total pixel count is small. The OR condition preserves these strokes by checking if their length is significant, even if their area is small.

#### **Q10: What is the physical meaning of a high excess kurtosis in a color channel?**
> **Answer:** High kurtosis (leptokurtic) means the pixel distribution has a sharp peak. This indicates that most pixels have almost identical values, representing uniform lighting and flat textures. Low kurtosis indicates a wide spread of colors and textures.

---

### Computer Science & Web Architecture Questions

#### **Q11: Why did you choose React + FastAPI over Python Gradio?**
> **Answer:** Gradio is useful for quick prototyping, but it doesn't allow for custom UI controls. React + FastAPI enables us to implement custom before/after sliders, real-time job status polling, and a clean, type-safe API contract using TypeScript and Pydantic.

#### **Q12: Why did you use daemon threads in `api/jobs.py`?**
> **Answer:** Processing high-resolution images takes 15–25 seconds. If we ran this inside the HTTP request thread, the browser connection would timeout. Spawning background threads (`daemon=True`) allows the API to return a job ID immediately while the image processes in the background.

#### **Q13: What is the purpose of the `_lock` in `api/jobs.py`?**
> **Answer:** Since Python background threads write to the shared `_jobs` dictionary while the main FastAPI thread reads from it, the lock (`with _lock:`) prevents data corruption and race conditions.

#### **Q14: Explain the role of the `_enhance_gate` Semaphore.**
> **Answer:** Real-ESRGAN uses a massive amount of RAM and CPU resources. If multiple users submitted images at the same time, running them concurrently would run the system Out of Memory. The Semaphore acts as a gate, ensuring that **only one image is enhanced at a time**.

#### **Q15: What is the purpose of `lru_cache` on PyTorch model loaders?**
> **Answer:** Loading a PyTorch model from disk and allocating its weights takes 2–3 seconds. Using `lru_cache` to keep the initialized model in memory saves this overhead.

#### **Q16: How does the U-Net architecture preserve spatial detail during downsampling?**
> **Answer:** U-Net uses **skip connections**. The encoder downsamples the image to capture context, which loses fine spatial details. The decoder reconstructs the image, and the skip connections copy high-resolution features from the encoder directly to the decoder at each scale, preserving thin character boundaries.

#### **Q17: What is the core difference between U-Net and DocEnTr?**
> **Answer:** U-Net is fully convolutional and processes local pixel neighborhoods. DocEnTr uses a Vision Transformer (ViT). It splits the image into $8\times8$ patches and uses self-attention to relate each patch to the rest of the image, capturing global page context.

#### **Q18: How does your image ID system handle duplicate filenames in different directories?**
> **Answer:** The system creates unique IDs by replacing folder separators with double underscores (e.g., `tamil_stone/image1.jpg` becomes `tamil_stone__image1.jpg`), preventing collisions.

#### **Q19: What is binary entropy and how do you use it as a confidence score?**
> **Answer:** Binary entropy measures the uncertainty of a probability value. If the network outputs $0.5$ (unsure), entropy is $1.0$ (maximum uncertainty). If it outputs $0.0$ or $1.0$, entropy is $0.0$ (perfect certainty). We calculate the mean entropy across all pixels and subtract it from 1 to get a confidence score.

#### **Q20: Why did you choose PNG for binarised images and JPEG for preprocessed images?**
> **Answer:** Preprocessed and enhanced images contain continuous color gradients, where JPEG's lossy compression saves space with minimal visual loss. Binarised images are strictly black-and-white mask boundaries. JPEG compression would introduce gray noise pixels around the edges of the text, so we use PNG (lossless compression) to preserve the sharp boundaries needed for OCR.

---

### Project Management & Process Questions (IEM)

#### **Q21: What is the bottleneck of this pipeline, and how can it be optimized?**
> **Answer:** The bottleneck is Stage 2 (Super-Resolution), taking 12–25 seconds per image on a CPU. This can be optimized by offloading the task to a GPU, which speeds up PyTorch inference by 5–8×, or by skipping super-resolution for high-contrast rubbings (using the pipeline's auto-routing).

#### **Q22: How does the pipeline ensure data integrity (non-destructive rule)?**
> **Answer:** The source scans in `data/raw/` are opened in read-only mode. Each pipeline stage writes its outputs to a separate, dedicated folder (`data/preprocessed/`, `data/enhanced/`, `data/binarised/`), ensuring raw research data is never modified or overwritten.

#### **Q23: How did you select your representative focus images?**
> **Answer:** We ran a dataset analysis script (`inspect_dataset.py`) to profile the images. We then hand-picked 11 images that represented extreme cases: lowest contrast, highest saturation, smallest resolution, and different scripts and materials, to benchmark the pipeline's robustness.

#### **Q24: What is the scope limit of Phase 1 of this project?**
> **Answer:** Phase 1 delivers up to Stage 4 (OCR & Transcription text extraction). Translation (Stage 5), structured record assembly (Stage 6), and PDF export are designed but deferred to Phase 2, allowing us to focus on building a highly accurate text extraction engine first.

#### **Q25: Why is Sauvola binarisation preferred over Otsu for stone inscriptions?**
> **Answer:** Otsu calculates a single global threshold for the entire image. Stone inscriptions often have uneven lighting and shadows, meaning a single threshold will wash out text in dark or bright regions. Sauvola calculates local thresholds, adapting to shadows and highlights across the stone surface.
