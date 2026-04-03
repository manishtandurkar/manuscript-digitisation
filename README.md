# Inscription Digitisation

A pipeline to preprocess, enhance, binarise, OCR, and assemble structured research records from images of inscriptions and manuscripts (stone, palm leaf, copper plate, paper, cave/rock paintings).

> NOTE: AGENTS.md is the single source of truth for this project. Read AGENTS.md fully before making changes.

---

## Table of Contents
- Project summary
- Scope & phases
- Implementation progress
- Quick start (Windows)
- Folder structure
- Installation
- Model weights & Tesseract
- Usage examples
- Pipeline stages overview
- Non-destructive rules
- Quality evaluation
- Running tests
- Contributing & coding conventions
- Troubleshooting
- References

---

## Project Summary

This repository implements an end-to-end pipeline to digitise historical inscriptions and manuscripts from South Asia. The goal is to convert scanned images into high-quality enhanced images, clean binarised images, transcriptions (OCR), and structured JSON research records.

---

## Scope & Phases

- **Phase 1 (current):** Preprocessing, Enhancement, Binarisation, OCR/Transcription, Record assembly. Translation (Phase 2) is designed but deferred.
- **Deliverable (Phase 1):** Robust OCR and transcription with quality evaluation.

---

## Implementation Progress (as of April 3, 2026)

### ✅ Completed

**1. Stage 1 — Preprocessing (`src/preprocess.py`)**
   - ✅ `load_image()` — loads BGR images with colour space preservation
   - ✅ `normalise_brightness()` — CLAHE histogram equalisation for uneven lighting
   - ✅ `auto_white_balance()` — grey-world assumption white balance
   - ✅ `deskew()` — Hough line transform rotation correction with safety limits (max 10° by default)
   - ✅ `crop_borders()` — removes blank/dark margins from scans using morphological operations
   - ✅ `preprocess()` — full chain orchestration with metadata preservation (EXIF, ICC profile, DPI)
   - ✅ `process_directory()` — batch preprocessing with pattern matching (e.g., "*.jpg")
   - ✅ CLI with argparse:
     - Single-image mode: `--input IMAGE_PATH --output OUTPUT_PATH`
     - Batch mode: `--input-dir DIR --output-dir DIR --pattern "*.jpg"`
     - Configurable deskew (--no-deskew, --deskew-min-lines, --deskew-max-angle)
     - Access copy control (--no-access-copy)
     - Logging levels (DEBUG, INFO, WARNING, ERROR)
   - ✅ Master + access copy strategy: TIFF master (lossless) + JPEG access (quality=95)
   - ✅ Comprehensive logging for all transformations with before/after dimensions
   - ✅ EXIF metadata sanitisation (removes problematic pointer tags)

**2. Utilities (`src/utils.py`)**
   - ✅ `read_image_metadata()` — extracts EXIF, ICC profile, DPI from images
   - ✅ `save_image()` — saves BGR images via PIL with metadata preservation
   - ✅ `_sanitise_exif()` — removes problematic EXIF pointer tags (34665, 34853, 40965)
   - ✅ `ensure_parent_dir()` — creates parent directories on demand

**3. Tests (`tests/test_preprocess.py`)**
   - ✅ `test_load_image_reads_sample()` — verifies image loading (3-channel BGR)
   - ✅ `test_crop_borders_removes_synthetic_frame()` — synthetic border removal test
   - ✅ `test_preprocess_writes_output()` — verifies both master and access copy creation
   - ✅ `test_preprocess_no_access_copy()` — access copy disabled mode
   - ✅ `test_preprocess_deskew_disabled()` — deskew disabled mode
   - ✅ `test_process_directory_writes_expected_output_name()` — batch processing output naming
   - ✅ All tests use temporary directories and real sample image (`data/raw/tamil_stone/IMG_3941.jpg`)
   - ✅ Can run with: `pytest -q` or `python -m unittest`

**4. Project Folder Structure**
   - ✅ `data/raw/` — read-only original scanned images
   - ✅ `data/enhanced/` — output of Stage 2 (enhancement)
   - ✅ `data/binarised/` — output of Stage 3 (binarisation)
   - ✅ `data/transcriptions/` — plain text OCR output
   - ✅ `data/translations/` — translated text (Phase 2)
   - ✅ `data/records/` — final assembled JSON records
   - ✅ `models/weights/` — directory for model weight files
   - ✅ `outputs/exports/` — PDF exports for download
   - ✅ `outputs/logs/` — processing logs
   - ✅ `src/__init__.py` — package marker
   - ✅ Sample image: `data/raw/tamil_stone/IMG_3941.jpg` (Tamil stone inscription)

**5. Documentation**
   - ✅ **AGENTS.md** — comprehensive project specification (15,000+ words)
     - End-to-end pipeline architecture
     - Detailed implementation specs for all 8 stages
     - Interdisciplinary team roles (CS/IT, ECE, IEM)
     - Quality evaluation criteria
     - Non-destructive processing rules
     - Known limitations and future work
   - ✅ **README.md** — this file (quick start, installation, usage guides)

---

### 🔄 In Progress / To-Do

**2. Stage 2 — Enhancement (`src/enhance.py`) — NEXT PRIORITY**
   - 🔲 `denoise()` — non-local means denoising (cv2.fastNlMeansDenoisingColored)
   - 🔲 `enhance_with_realesrgan()` — Real-ESRGAN super-resolution (2x/4x)
   - 🔲 `dstretch()` — decorrelation stretch for faded pigment (DStretch algorithm by Jon Harman)
   - 🔲 `sharpen()` — unsharp mask sharpening for crisp character edges
   - 🔲 `enhance()` — full pipeline orchestration
   - 🔲 Model weight loading and GPU detection (CUDA/CPU)
   - 🔲 Batch enhancement with tile-based processing (prevents OOM on large images)
   - 🔲 CLI with argparse for --input, --output, --model selection, --use-dstretch
   - 🔲 Comprehensive tests

**3. Stage 3 — Binarisation (`src/binarise.py`)**
   - 🔲 `binarise_sauvola()` — Sauvola local thresholding (PREFERRED for inscriptions)
   - 🔲 `binarise_otsu()` — Otsu global thresholding (fast, for clean paper)
   - 🔲 `binarise_adaptive()` — OpenCV adaptive mean thresholding (fallback)
   - 🔲 `remove_noise_blobs()` — morphological noise removal (size-based filtering)
   - 🔲 Morphological closing to reconnect broken character strokes
   - 🔲 `binarise()` — stage orchestration with method selection
   - 🔲 CLI and comprehensive tests

**4. Stage 4 — OCR & Transcription (`src/ocr.py`)**
   - 🔲 Script detection (heuristic or ML-based) — returns script name string
   - 🔲 Tesseract integration:
     - Tamil (tam), Sanskrit (san), Kannada (kan), Telugu (tel), Malayalam (mal), Hindi (hin)
     - Config: `--oem 1 --psm 6` (LSTM engine, uniform block)
   - 🔲 EasyOCR integration with multi-language fallback
   - 🔲 Ensemble method — run both engines, merge by confidence score
   - 🔲 Confidence thresholding:
     - ≥ 0.85 → verified
     - 0.60–0.84 → review needed
     - < 0.60 → uncertain (flagged for manual review)
   - 🔲 Line/word bounding box extraction
   - 🔲 Output schema:
     ```json
     {
       "script": "tamil",
       "text": "...",
       "lines": [{line_number, text, confidence, bounding_box, uncertain}],
       "overall_confidence": 0.87,
       "engine_used": "tesseract+easyocr ensemble",
       "uncertain_regions": [[x1, y1, x2, y2]]
     }
     ```
   - 🔲 CLI and tests

**5. Stage 5 — Translation (`src/translate.py`) — PHASE 2 (time-permitting)**
   - 🔲 Helsinki-NLP OPUS-MT model integration (for post-10th century texts)
   - 🔲 LLM fallback — Claude/GPT-4 API integration (for ancient/ambiguous texts)
   - 🔲 Modern language form generation (e.g., classical Tamil → modern Tamil)
   - 🔲 Translator notes for ambiguous segments
   - 🔲 Uncertain segment handling with explanatory notes
   - 🔲 Tests
   - ℹ️ **Status:** Fully designed in AGENTS.md § 5, deferred per mentor guidance (implement only if Phase 1 completes early)

**6. Stage 6 — Record Assembly (`src/record.py`)**
   - 🔲 `generate_record_id()` — auto-generate INS-YYYY-NNNN sequential IDs
   - 🔲 `assemble_record()` — bundle all outputs (image, transcription, translation, metadata)
   - 🔲 `save_record()` — write record to JSON disk
   - 🔲 `export_pdf()` — researcher-friendly PDF with:
     - Side-by-side before/after images
     - Transcription with confidence highlights
     - Translation and translator notes
     - Metadata table (artefact type, location, condition, etc.)
     - Citation block
   - 🔲 Full record JSON schema implementation
   - 🔲 Tests

**7. Stage 7 — Pipeline Orchestration (`src/pipeline.py`)**
   - 🔲 `process_single()` — end-to-end pipeline for one image
   - 🔲 `process_batch()` — batch processing with multiprocessing.Pool (configurable workers)
   - 🔲 Progress tracking and logging
   - 🔲 Error recovery (fail-safe continuation on per-image errors)
   - 🔲 CSV metadata integration (batch_meta.csv)
   - 🔲 Comprehensive tests

**8. Stage 8 — Gradio UI (`app.py`)**
   - 🔲 "Process new image" tab:
     - Image upload widget
     - Artefact metadata form (type, location, script, period, material, condition)
     - Run button → pipeline execution → results display
   - 🔲 "Browse records" tab with search/filter by script, location, period
   - 🔲 "View record" detail card with before/after images, transcription
   - 🔲 "Export" button (PDF/JSON download)
   - 🔲 "Translation" tab (DISABLED in Phase 1, activated in Phase 2)
   - 🔲 Error handling and status messages
   - 🔲 Tests

**9. Optional REST API (`api.py`)**
   - 🔲 FastAPI endpoints:
     - POST `/process` — submit image + metadata
     - GET `/records/{record_id}` — fetch record
     - GET `/records?script=tamil&period=7th` — search/filter
     - GET `/exports/{record_id}` — download PDF/JSON
   - 🔲 Async processing with background tasks
   - 🔲 OpenAPI docs/Swagger UI
   - 🔲 Tests

**10. Quality Evaluation Metrics (`src/metrics.py` — ECE responsibility)**
   - 🔲 PSNR (Peak Signal-to-Noise Ratio) — target ≥ 30 dB
   - 🔲 SSIM (Structural Similarity Index) — target ≥ 0.85
   - 🔲 CNR (Contrast-to-Noise Ratio) — particularly useful for low-contrast inscriptions
   - 🔲 Sharpness score (Laplacian variance)
   - 🔲 `full_quality_report()` — consolidated metrics dict
   - 🔲 Integration into pipeline.py so every record logs quality scores

**11. Custom Filters & Signal Processing (`src/filters.py`, `src/analysis.py` — ECE responsibility)**
   - 🔲 Gabor filter bank — separate text texture from background
   - 🔲 Directional edge enhancement — useful for carved inscriptions
   - 🔲 FFT-based periodic noise removal — remove scanner line artifacts
   - 🔲 Colour distribution analysis — per-channel statistics
   - 🔲 Histogram comparison plots (before/after enhancement)

**12. Dependencies & Environment**
   - 🔲 Create `requirements.txt` with pinned versions (see AGENTS.md § 4)
   - 🔲 Create `environment.yml` for Conda users
   - 🔲 Add GPU detection and optional CUDA setup guide

---

## Sample Image

A test image is included at `data/raw/tamil_stone/IMG_3941.jpg` for development and testing. This is a real Tamil stone inscription.

---

## Quick Start (Windows PowerShell)

1. Clone the repo and change to project root:

   ```powershell
   git clone <repo-url>
   cd path\to\project
   ```

2. Create and activate a virtual environment (recommended):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install --upgrade pip
   ```

3. Install dependencies (once requirements.txt is created):

   ```powershell
   pip install -r requirements.txt
   ```

4. Install Tesseract (see "Tesseract" section below) and download model weights.

5. Run preprocessing on the sample image:

   ```powershell
   python -m src.preprocess --input data\raw\tamil_stone\IMG_3941.jpg --output data\enhanced\IMG_3941_preprocessed.tif
   ```

6. View the preprocessed output in `data/enhanced/`.

7. (Future) Run the Gradio UI once app.py is ready:

   ```powershell
   python app.py
   # open http://localhost:7860 in your browser
   ```

---

## Folder Structure (Key Paths)

```
C:\Projects\IDP\Project\
├── AGENTS.md                          (← read this first: project spec)
├── README.md                          (← you are here)
├── requirements.txt                   (← dependencies; to be created)
│
├── data\
│   ├── raw\                           (read-only: original scans)
│   │   └── tamil_stone\
│   │       └── IMG_3941.jpg           (← sample test image)
│   ├── enhanced\                      (output of Stage 2)
│   ├── binarised\                     (output of Stage 3)
│   ├── transcriptions\                (output of Stage 4)
│   ├── translations\                  (output of Stage 5; Phase 2)
│   └── records\                       (final JSON records)
│
├── models\
│   ├── weights\                       (model files: RealESRGAN, etc.)
│   └── custom\                        (fine-tuned models; future)
│
├── src\
│   ├── __init__.py
│   ├── preprocess.py                  (✅ Stage 1: implemented)
│   ├── enhance.py                     (🔲 Stage 2)
│   ├── binarise.py                    (🔲 Stage 3)
│   ├── ocr.py                         (🔲 Stage 4)
│   ├── translate.py                   (🔲 Stage 5; Phase 2)
│   ├── record.py                      (🔲 Stage 6)
│   ├── pipeline.py                    (🔲 Stage 7)
│   ├── utils.py                       (✅ utilities)
│   ├── filters.py                     (🔲 custom filters; ECE)
│   ├── analysis.py                    (🔲 analysis tools; ECE)
│   └── metrics.py                     (🔲 quality metrics; ECE)
│
├── app.py                             (🔲 Stage 8: Gradio UI)
├── api.py                             (🔲 optional: FastAPI REST)
│
├── tests\
│   ├── __init__.py
│   ├── test_preprocess.py             (✅ 6 tests passing)
│   ├── test_enhance.py                (🔲 to be written)
│   ├── test_binarise.py               (🔲 to be written)
│   ├── test_ocr.py                    (🔲 to be written)
│   └── sample_images\                 (← test fixtures)
│
└── outputs\
    ├── exports\                       (PDF downloads)
    └── logs\                          (processing logs)
```

---

## Installation (Detailed)

### Prerequisites
- Python 3.10 (recommended) or compatible 3.9/3.11
- pip (or conda)
- (Optional) GPU + CUDA for faster Real-ESRGAN

### Install Python Dependencies

Once `requirements.txt` is available:

```powershell
pip install -r requirements.txt
```

### Tesseract (Windows)

**Option A — Chocolatey (recommended for automation):**

```powershell
choco install -y tesseract
```

**Option B — Manual installer:**
1. Download the Windows installer (e.g., UB Mannheim build) from [Tesseract releases](https://github.com/UB-Mannheim/tesseract/wiki)
2. Install and note the installation folder (e.g., `C:\Program Files\Tesseract-OCR`)
3. Download tessdata language packs (tam, san, kan, tel, mal, hin) into the `tessdata` folder
4. Add the installation folder to your PATH

**Set TESSDATA_PREFIX (if necessary):**

```powershell
$env:TESSDATA_PREFIX = "C:\Program Files\Tesseract-OCR\tessdata"
```

**Verify installation:**

```powershell
tesseract --version
python -c "import pytesseract; print(pytesseract.get_languages())"
```

### Model Weights (Real-ESRGAN)

Create the weights folder:

```powershell
New-Item -ItemType Directory -Force -Path models\weights
```

Download recommended weights using curl:

```powershell
curl -L -o models\weights\RealESRGAN_x4plus.pth `
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

curl -L -o models\weights\RealESRGAN_x4plus_anime_6B.pth `
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth
```

Or use a PowerShell script:

```powershell
$urls = @(
  "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
  "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
)
foreach ($url in $urls) {
  $filename = $url.Split('/')[-1]
  Invoke-WebRequest -Uri $url -OutFile "models\weights\$filename"
}
```

---

## Usage Examples

### 1. Single-Image Preprocessing

```powershell
python -m src.preprocess `
  --input data\raw\tamil_stone\IMG_3941.jpg `
  --output data\enhanced\IMG_3941_preprocessed.tif
```

### 2. Batch Preprocessing

```powershell
python -m src.preprocess `
  --input-dir data\raw\tamil_stone `
  --output-dir data\enhanced `
  --pattern "*.jpg" `
  --no-access-copy
```

### 3. Preprocessing with Custom Deskew Settings

```powershell
python -m src.preprocess `
  --input data\raw\tamil_stone\IMG_3941.jpg `
  --output data\enhanced\IMG_3941_preprocessed.tif `
  --deskew-min-lines 15 `
  --deskew-max-angle 5.0 `
  --log-level DEBUG
```

### 4. Disable Deskewing (if images are over-corrected)

```powershell
python -m src.preprocess `
  --input data\raw\tamil_stone\IMG_3941.jpg `
  --output data\enhanced\IMG_3941_preprocessed.tif `
  --no-deskew
```

### 5. Python API (for scripting)

```python
from src.preprocess import preprocess, process_directory

# Single image
preprocessed = preprocess(
    r"data\raw\tamil_stone\IMG_3941.jpg",
    r"data\enhanced\IMG_3941_preprocessed.tif",
    enable_deskew=True,
    save_access_copy=True
)

# Batch processing
output_paths = process_directory(
    r"data\raw\tamil_stone",
    r"data\enhanced",
    pattern="*.jpg",
    enable_deskew=True,
    save_access_copy=True
)
```

### 6. Gradio UI (once app.py is ready)

```powershell
python app.py
# open http://localhost:7860 in your browser
```

---

## Pipeline Stages (Overview)

| Stage | File | Status | Description |
|-------|------|--------|-------------|
| 1 | `src/preprocess.py` | ✅ | Normalise brightness, white balance, deskew, crop borders |
| 2 | `src/enhance.py` | 🔲 | Denoise, Real-ESRGAN, DStretch, sharpen |
| 3 | `src/binarise.py` | 🔲 | Sauvola/Otsu thresholding, morphological ops, noise removal |
| 4 | `src/ocr.py` | 🔲 | Script detection, Tesseract+EasyOCR ensemble, confidence scoring |
| 5 | `src/translate.py` | 🔲 | MT + LLM-based translation (Phase 2) |
| 6 | `src/record.py` | 🔲 | Assemble JSON records, export PDFs |
| 7 | `src/pipeline.py` | 🔲 | Orchestrate all stages, batch processing |
| 8 | `app.py` | 🔲 | Gradio UI for upload, browse, export |

---

## Non-Destructive Processing Rules (MANDATORY)

1. **Never overwrite original files.** Raw images in `data/raw/` are read-only. All outputs go to `data/enhanced/`, `data/binarised/`, etc.
2. **Always save both master and access copies:**
   - Master: TIFF lossless (for archival)
   - Access: JPEG quality=95 (for web/preview)
3. **Follow the 3-2-1 backup rule:**
   - 3 copies of original data
   - 2 different storage media
   - 1 offsite / cloud backup
4. **Preserve EXIF metadata** through all processing stages using PIL/piexif.
5. **Log every operation** — what was done, when, by which pipeline version.
6. **Version control model weights** — record which model version was used in each record's processing_log.

---

## Quality Evaluation

Recommended minimum thresholds for processed images:

| Metric | Target | Tool |
|--------|--------|------|
| PSNR | ≥ 30 dB | `skimage.metrics.peak_signal_noise_ratio` |
| SSIM | ≥ 0.85 | `skimage.metrics.structural_similarity` |
| OCR avg confidence | ≥ 0.70 | Built into Tesseract / EasyOCR |
| Character Error Rate (CER) | ≤ 0.15 | `jiwer` library |

---

## Running Tests

Run all unit tests:

```powershell
pip install pytest
pytest -q
```

Or using unittest directly:

```powershell
python -m unittest discover tests -v
```

Run a specific test file:

```powershell
pytest tests/test_preprocess.py -v
```

Run a specific test:

```powershell
pytest tests/test_preprocess.py::PreprocessTests::test_load_image_reads_sample -v
```

---

## Contributing & Coding Conventions

1. **Read AGENTS.md first** — it is the project's single source of truth.
2. **Keep changes focused.** Avoid touching unrelated files.
3. **Write tests for new functionality** and run existing tests before pushing.
4. **Follow the commit trailer rule** (include the required Co-authored-by trailer):
   ```
   Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
   ```
5. **Use type hints** in function signatures (PEP 484).
6. **Comment only when necessary** — code should be self-explanatory.
7. **Log important operations** using Python's `logging` module (not print statements).

---

## Troubleshooting

**Q: Tesseract returns no languages or missing language packs**

A: Ensure tessdata contains the required language files and `TESSDATA_PREFIX` or `PATH` is set:

```powershell
$env:TESSDATA_PREFIX = "C:\Program Files\Tesseract-OCR\tessdata"
python -c "import pytesseract; print(pytesseract.get_languages())"
```

**Q: Real-ESRGAN runs out of GPU memory (OOM)**

A: Reduce tile size or run on CPU; increase `tile`/`tile_pad` params in enhance.py.

**Q: Low OCR confidence on preprocessing output**

A: Experiment with different binarisation methods (Sauvola window size), or use ensemble with EasyOCR.

**Q: Deskew is over-correcting and tilting images**

A: Run with `--no-deskew` or adjust `--deskew-max-angle` to a smaller value (e.g., 5.0 instead of 10.0).

**Q: How do I run preprocessing in debug mode?**

A:

```powershell
python -m src.preprocess --input ... --output ... --log-level DEBUG
```

---

## References & Further Reading

- **DStretch algorithm** — Jon Harman (dstretch.com)
- **Real-ESRGAN** — https://github.com/xinntao/Real-ESRGAN
- **Tesseract OCR** — https://github.com/tesseract-ocr/tesseract
- **AGENTS.md** — canonical project instructions and implementation details
- **Indiscapes dataset** — IIIT Hyderabad (ihdia.iiit.ac.in)
- **Brahmi OCR research** — arxiv.org/abs/2501.01981
- **Tamil NLP resources** — ai4bharat.org

---

## Contact & License

- **Project maintainers:** See repository owners and AGENTS.md
- **License:** See LICENSE in the repository root. If no LICENSE is present, contact repository maintainers.

---

*Last updated: April 3, 2026*  
*For the latest implementation status, see "Implementation Progress" section above.*
