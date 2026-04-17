# AGENTS.md — Inscription & Manuscript Digitisation Project

> This file is the single source of truth for any AI agent or developer working on this project.
> Read it fully before writing a single line of code.

---

## 1. Project Overview

This project digitises, enhances, transcribes, and translates historical and archaeological
inscriptions and manuscripts from South Asia. The input is scanned image files of physical
artefacts. The output is a structured research record per artefact containing: an enhanced
image, a transcription in the original script, an English translation, and rich metadata.

### Goal statement

> Take unclear, degraded images of stone inscriptions, palm leaf manuscripts, copper plates,
> paper manuscripts, and cave/rock paintings — and produce clean, readable, searchable,
> citable digital records for researchers, historians, linguists, and the public.

### Scope note (updated March 2026)

> **Current phase (Phase 1) delivers up to Stage 4 — OCR & Transcription.**
> Translation (Stage 5) is architecturally designed and model-selected, but implementation
> is deferred to Phase 2 and will be undertaken only if the project timeline permits.
> The mentor has advised delivering a robust, well-evaluated text extraction system
> before proceeding to translation.

### End-to-end pipeline (in order)

```
Raw image input
      ↓
1. Preprocessing        (normalise, white balance, crop, orient)          ← Phase 1
      ↓
2. Enhancement          (denoise, sharpen, super-resolution, DStretch)     ← Phase 1
      ↓
3. Binarisation         (separate text from background)                    ← Phase 1
      ↓
4. OCR / Transcription  (extract characters as Unicode text)               ← Phase 1 ENDPOINT
      ↓
5. Translation          (convert to modern English via NLP/LLM)            ← Phase 2 (time-permitting)
      ↓
6. Record assembly      (bundle image + text + metadata into a structured record)
      ↓
7. Storage & export     (save to database, export PDF/JSON)
      ↓
8. UI / portal          (Gradio interface for upload, browse, search)
```

---

## 2. Supported Artefact Types

The pipeline must handle ALL of the following input types:

| Type | Key challenges | Special tool |
|---|---|---|
| Stone inscriptions | Low contrast, surface weathering, shadow | DStretch colour enhancement |
| Palm leaf manuscripts | Fragile, yellowed, ink fading | Binarisation + contrast stretch |
| Copper plate inscriptions | Reflective surface, oxidation | HDR normalisation |
| Paper manuscripts | Foxing, stains, torn edges | Inpainting + denoising |
| Cave / rock paintings | Uneven lighting, rough texture | DStretch + shadow removal |

---

## 3. Folder Structure

```
inscription-digitisation/
├── AGENTS.md                    ← you are here
├── README.md
├── requirements.txt
├── environment.yml
│
├── data/
│   ├── raw/                     ← original scanned images (never modify these)
│   ├── enhanced/                ← output of enhancement stage
│   ├── binarised/               ← output of binarisation stage
│   ├── transcriptions/          ← plain text OCR output per image
│   ├── translations/            ← translated text per image
│   └── records/                 ← final assembled JSON records
│
├── models/
│   ├── weights/                 ← model weight files (.pth)
│   │   ├── RealESRGAN_x4plus.pth
│   │   └── RealESRGAN_x4plus_anime_6B.pth
│   └── custom/                  ← fine-tuned models (future)
│
├── src/
│   ├── preprocess.py            ← Stage 1: preprocessing
│   ├── enhance.py               ← Stage 2: AI enhancement
│   ├── binarise.py              ← Stage 3: binarisation
│   ├── ocr.py                   ← Stage 4: OCR + transcription
│   ├── translate.py             ← Stage 5: translation
│   ├── record.py                ← Stage 6: record assembly
│   ├── pipeline.py              ← orchestrates all stages end-to-end
│   └── utils.py                 ← shared helpers
│
├── app.py                       ← Gradio web interface
├── api.py                       ← optional REST API (FastAPI)
│
├── tests/
│   ├── test_enhance.py
│   ├── test_ocr.py
│   └── sample_images/           ← small test images for CI
│
└── outputs/
    ├── exports/                 ← PDF exports for download
    └── logs/                    ← processing logs
```

---

## 4. Environment Setup

### Step 1 — Clone and create environment

```bash
git clone <repo-url>
cd inscription-digitisation

conda create -n inscriptions python=3.10
conda activate inscriptions
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### requirements.txt (full)

```
# Image processing
opencv-python==4.9.0.80
Pillow==10.3.0
numpy==1.26.4
scikit-image==0.23.2
imageio==2.34.1

# AI enhancement
basicsr==1.4.2
realesrgan==0.3.0
torch>=2.0.0
torchvision>=0.15.0

# OCR
pytesseract==0.3.10
easyocr==1.7.1
indic-transliteration==2.3.57

# NLP / translation
transformers==4.40.0
sentencepiece==0.2.0
sacremoses==0.1.1

# Web UI
gradio==4.31.0

# API
fastapi==0.111.0
uvicorn==0.29.0

# Storage / export
sqlalchemy==2.0.30
fpdf2==2.7.9
tinydb==4.8.0

# Utilities
tqdm==4.66.4
python-dotenv==1.0.1
loguru==0.7.2
pydantic==2.7.1
```

### Step 3 — Download model weights

```bash
mkdir -p models/weights
cd models/weights

# Real-ESRGAN general model (best for inscriptions)
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

# Real-ESRGAN anime model (better for line-art style carvings)
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth

cd ../..
```

### Step 4 — Install Tesseract with Indic language packs

```bash
# Ubuntu / Debian
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-tam   # Tamil
sudo apt-get install tesseract-ocr-san   # Sanskrit
sudo apt-get install tesseract-ocr-kan   # Kannada
sudo apt-get install tesseract-ocr-tel   # Telugu
sudo apt-get install tesseract-ocr-mal   # Malayalam
sudo apt-get install tesseract-ocr-hin   # Hindi (Devanagari)

# macOS
brew install tesseract
brew install tesseract-lang
```

### Step 5 — Verify setup

```bash
python -c "import cv2, PIL, torch, easyocr; print('All core imports OK')"
python -c "import pytesseract; print(pytesseract.get_languages())"
```

---

## 5. Stage-by-Stage Implementation

### Stage 1 — Preprocessing (`src/preprocess.py`)

**Purpose:** Normalise raw images before AI enhancement. Fix orientation, colour, and crop.

**Key functions:**

```python
def load_image(path: str) -> np.ndarray:
    """Load image as BGR numpy array, applying EXIF orientation via PIL so output is visually upright."""

def normalise_brightness(img: np.ndarray) -> np.ndarray:
    """CLAHE histogram equalisation for uneven lighting."""

def auto_white_balance(img: np.ndarray) -> np.ndarray:
    """Grey-world assumption white balance correction."""

def crop_borders(img: np.ndarray, threshold: int = 10) -> np.ndarray:
    """Remove blank/dark border margins from scans."""

def preprocess(img_path: str, output_path: str) -> np.ndarray:
    """Run full preprocessing chain and save result as JPEG."""

def build_output_path(input_path, output_dir) -> Path:
    """Returns output_dir / {stem}_preprocessed.jpg"""

def process_directory(input_dir, output_dir, pattern="*.jpg") -> list[Path]:
    """Batch preprocess all matching images in a directory."""
```

**Implementation notes:**
- Use `PIL.ImageOps.exif_transpose` in `load_image` to correct orientation before any processing — EXIF orientation tag is NOT carried over to output (already baked into pixels)
- Use `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))` for CLAHE
- Output format is **JPEG** (quality=95) — no TIFF, no access copies
- Log before/after dimensions for every image

---

### Stage 2 — Enhancement (`src/enhance.py`)

**Purpose:** Apply AI-based denoising, sharpening, and super-resolution. This is the core
value of the project — making previously unreadable inscriptions legible.

**Key functions to implement:**

```python
def denoise(img: np.ndarray, strength: int = 10) -> np.ndarray:
    """Non-local means denoising. strength=10 for mild, 20 for heavy damage."""

def enhance_with_realesrgan(img: np.ndarray, scale: int = 2,
                             model_path: str = "models/weights/RealESRGAN_x4plus.pth"
                             ) -> np.ndarray:
    """Super-resolution + sharpening via Real-ESRGAN."""

def dstretch(img: np.ndarray, colour_space: str = "LAB") -> np.ndarray:
    """DStretch colour decorrelation stretch for revealing faded pigment.
    Especially powerful for cave paintings and faded stone inscriptions.
    colour_space options: LAB, YDS, YBK, LDS"""

def sharpen(img: np.ndarray, amount: float = 1.5) -> np.ndarray:
    """Unsharp mask sharpening to crisp up character edges."""

def enhance(img_path: str, output_path: str,
            use_dstretch: bool = False) -> np.ndarray:
    """Full enhancement chain. Set use_dstretch=True for cave/rock paintings."""
```

**Real-ESRGAN setup:**

```python
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

def build_upsampler(model_path: str, scale: int = 4) -> RealESRGANer:
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=scale)
    return RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=400,          # tile size for large images (prevents OOM)
        tile_pad=10,
        pre_pad=0,
        half=False         # set True if using GPU with fp16 support
    )
```

**DStretch implementation:**

```python
import numpy as np

def dstretch(img: np.ndarray, colour_space: str = "LAB") -> np.ndarray:
    """
    Decorrelation stretch. Computes covariance of colour channels,
    then transforms to remove correlation — revealing colour differences
    invisible to the human eye. Original algorithm by Jon Harman.
    """
    img_float = img.astype(np.float64) / 255.0
    flat = img_float.reshape(-1, 3)
    mean = flat.mean(axis=0)
    centered = flat - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    stretch_matrix = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues + 1e-10)) @ eigenvectors.T
    stretched = centered @ stretch_matrix.T
    stretched = (stretched - stretched.min()) / (stretched.max() - stretched.min() + 1e-10)
    return (stretched.reshape(img_float.shape) * 255).astype(np.uint8)
```

**Implementation notes:**
- Always save both the original and enhanced image — never overwrite the original
- Use `outscale=2` for Real-ESRGAN (4x model but 2x output) — avoids over-smoothing
- For stone inscriptions: denoise → Real-ESRGAN → sharpen
- For cave paintings: denoise → DStretch → sharpen
- For palm leaf: auto_white_balance → denoise → Real-ESRGAN

---

### Stage 3 — Binarisation (`src/binarise.py`)

**Purpose:** Convert enhanced image to clean black-and-white so OCR can extract characters.

**Key functions to implement:**

```python
def binarise_sauvola(img: np.ndarray, window_size: int = 25) -> np.ndarray:
    """Sauvola local thresholding — best for uneven backgrounds (stone, palm leaf)."""

def binarise_otsu(img: np.ndarray) -> np.ndarray:
    """Otsu global thresholding — fast, good for clean paper manuscripts."""

def binarise_adaptive(img: np.ndarray) -> np.ndarray:
    """OpenCV adaptive mean thresholding — fallback for mixed quality images."""

def remove_noise_blobs(binary: np.ndarray, min_size: int = 50) -> np.ndarray:
    """Remove small disconnected components (dust, noise) from binary image."""

def binarise(img_path: str, output_path: str,
             method: str = "sauvola") -> np.ndarray:
    """Binarise image. method: 'sauvola' | 'otsu' | 'adaptive'"""
```

**Implementation notes:**
- Always convert to greyscale before binarisation
- Sauvola is preferred for most inscription types
- After binarisation, apply morphological closing to connect broken character strokes:
  `cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)` with a 2x2 kernel
- Save binarised image as PNG (lossless)

---

### Stage 4 — OCR & Transcription (`src/ocr.py`)

**Purpose:** Extract text characters from the binarised image as Unicode text.

**Script detection and routing:**

```python
SCRIPT_CONFIG = {
    "tamil":      {"tesseract_lang": "tam",  "easyocr_lang": ["ta"]},
    "sanskrit":   {"tesseract_lang": "san",  "easyocr_lang": ["hi"]},  # fallback via Devanagari
    "kannada":    {"tesseract_lang": "kan",  "easyocr_lang": ["kn"]},
    "telugu":     {"tesseract_lang": "tel",  "easyocr_lang": ["te"]},
    "malayalam":  {"tesseract_lang": "mal",  "easyocr_lang": ["ml"]},
    "devanagari": {"tesseract_lang": "hin",  "easyocr_lang": ["hi"]},
    "brahmi":     {"tesseract_lang": None,   "easyocr_lang": None},    # custom model needed
    "grantha":    {"tesseract_lang": None,   "easyocr_lang": None},    # custom model needed
}
```

**Key functions to implement:**

```python
def detect_script(img: np.ndarray) -> str:
    """Heuristic or model-based script detection. Returns script name string."""

def ocr_tesseract(img: np.ndarray, lang: str) -> dict:
    """Run Tesseract OCR. Returns {text, confidence, word_boxes}."""

def ocr_easyocr(img: np.ndarray, langs: list) -> dict:
    """Run EasyOCR. Returns {text, confidence, word_boxes}."""

def ocr_ensemble(img: np.ndarray, script: str) -> dict:
    """Run both engines and merge results by confidence score."""

def transcribe(img_path: str, script: str = "auto",
               output_path: str = None) -> dict:
    """Full transcription pipeline. Returns structured transcription dict."""
```

**Transcription output schema:**

```python
{
    "script": "tamil",
    "text": "கஞ்சி மாநகர் பல்லவ குல தீபம்",
    "lines": [
        {
            "line_number": 1,
            "text": "கஞ்சி மாநகர்",
            "confidence": 0.91,
            "bounding_box": [x, y, w, h],
            "uncertain": False
        }
    ],
    "overall_confidence": 0.87,
    "engine_used": "tesseract+easyocr ensemble",
    "uncertain_regions": [[x1, y1, x2, y2]]   # regions with confidence < 0.6
}
```

**Confidence thresholds:**
- >= 0.85 → mark as verified
- 0.60–0.84 → mark as review needed
- < 0.60 → mark as uncertain (highlight in output)

**Implementation notes:**
- For Brahmi and Grantha scripts, OCR will initially return empty — flag for manual transcription
- Always save raw OCR output alongside cleaned output for debugging
- Tesseract config: `--oem 1 --psm 6` (LSTM engine, uniform block of text)
- EasyOCR: `detail=1` to get bounding boxes per word

---

### Stage 5 — Translation (`src/translate.py`) — PHASE 2 (time-permitting)

> **This stage is NOT in scope for the current phase.** The architecture, model selection,
> and output schema are fully designed below for Phase 2 implementation.
> In the current phase, the translation field in every record is left blank and marked
> "Phase 2 — pending". The record schema already includes the translation field so
> no schema change will be required when Phase 2 is implemented.

**Purpose:** Convert transcribed ancient script text to modern English (and optionally
to the modern form of the same language — e.g., classical Tamil → modern Tamil).

**Models to use:**

```python
TRANSLATION_MODELS = {
    # Hugging Face model IDs
    "tamil_to_english":     "Helsinki-NLP/opus-mt-dra-en",
    "hindi_to_english":     "Helsinki-NLP/opus-mt-hi-en",
    "kannada_to_english":   "Helsinki-NLP/opus-mt-dra-en",
    "telugu_to_english":    "Helsinki-NLP/opus-mt-dra-en",
    "malayalam_to_english": "Helsinki-NLP/opus-mt-dra-en",
    # For ancient/classical texts, use LLM fallback (see below)
}
```

**Key functions to implement:**

```python
def translate_with_model(text: str, source_lang: str,
                          target_lang: str = "en") -> dict:
    """Translate using Helsinki-NLP opus-mt models via Hugging Face transformers."""

def translate_with_llm(text: str, script: str, context: str = "") -> dict:
    """
    Fallback for ancient/damaged/ambiguous text using an LLM (e.g. GPT-4 or Claude API).
    Provide context about the artefact type and period for better results.
    Returns translation with explanatory notes on ambiguous segments.
    """

def translate(transcription: dict, artefact_context: dict = None) -> dict:
    """
    Full translation pipeline. Tries model first, falls back to LLM for
    low-confidence or ancient-script text.
    """
```

**Translation output schema:**

```python
{
    "source_script": "tamil",
    "source_text": "கஞ்சி மாநகர் பல்லவ குல தீபம்",
    "translation": "The great city of Kanchi, lamp of the Pallava lineage",
    "modern_source": "காஞ்சி மாநகர் பல்லவ குல விளக்கு",  # modern Tamil (optional)
    "confidence": 0.84,
    "method": "opus-mt",
    "notes": [
        {"segment": "தீபம்", "note": "Literally 'lamp' — used metaphorically as 'glory' or 'light'"}
    ],
    "uncertain_segments": ["மகேந்திர [?]"]
}
```

**Implementation notes:**
- For ancient texts (pre-10th century), always use LLM fallback — MT models are not trained on archaic forms
- Always include translator notes for terms that have multiple valid interpretations
- Preserve untranslatable proper nouns (place names, personal names) in original script + transliteration
- Use `indic-transliteration` library for romanisation of untranslatable segments

---

### Stage 6 — Record Assembly (`src/record.py`)

**Purpose:** Bundle all outputs (enhanced image, transcription, translation, metadata)
into a single structured research record.

**Record schema (JSON):**

```python
{
    "record_id": "INS-2024-0047",           # auto-generated: INS-{year}-{sequence}
    "created_at": "2024-03-20T10:30:00Z",
    "status": "verified",                   # draft | review | verified

    "artefact": {
        "type": "stone_inscription",        # stone | palm_leaf | copper_plate | paper | cave
        "material": "granite",
        "period_estimate": "7th century CE",
        "dynasty": "Pallava",
        "location": {
            "site": "Mamallapuram",
            "district": "Chengalpattu",
            "state": "Tamil Nadu",
            "country": "India",
            "coordinates": [12.6269, 80.1927]  # [lat, lng] if available
        },
        "dimensions_cm": {"height": 45, "width": 30},
        "condition": "partial_damage",      # good | partial_damage | heavy_damage | fragmentary
        "collection": "Tamil Nadu State Archives",
        "accession_number": "TNSA-1987-0234"
    },

    "images": {
        "original": "data/raw/INS-2024-0047_original.tif",
        "enhanced": "data/enhanced/INS-2024-0047_enhanced.tif",
        "binarised": "data/binarised/INS-2024-0047_binarised.png",
        "thumbnail": "data/enhanced/INS-2024-0047_thumb.jpg",
        "enhancement_method": "RealESRGAN_x4plus + OpenCV denoising",
        "processed_at": "2024-03-20T10:28:00Z"
    },

    "transcription": {
        "script": "Tamil-Grantha",
        "language": "Classical Tamil",
        "text": "கஞ்சி மாநகர் பல்லவ குல தீபம் மகேந்திர [?] வர்மன்",
        "lines": [...],                     # per-line breakdown from OCR stage
        "overall_confidence": 0.87,
        "uncertain_segments": ["மகேந்திர [?]"],
        "ocr_engine": "tesseract+easyocr ensemble"
    },

    "translation": {
        "english": null,                    // null in Phase 1; populated in Phase 2
        "modern_source_language": null,
        "confidence": null,
        "method": null,
        "notes": [],
        "status": "phase_2_pending"         // field reserved; not yet implemented
    },

    "citation": {
        "suggested_cite": "Inscription INS-2024-0047. Mamallapuram, Tamil Nadu. 7th century CE. Processed by [Project Name], March 2024.",
        "doi": null,
        "licence": "CC BY 4.0"
    },

    "processing_log": [
        {"stage": "preprocess", "status": "success", "duration_s": 1.2},
        {"stage": "enhance",    "status": "success", "duration_s": 14.7},
        {"stage": "binarise",   "status": "success", "duration_s": 0.4},
        {"stage": "ocr",        "status": "success", "duration_s": 3.1},
        {"stage": "translate",  "status": "success", "duration_s": 2.8}
    ]
}
```

**Key functions to implement:**

```python
def generate_record_id(year: int = None) -> str:
    """Auto-generate sequential record ID: INS-{YYYY}-{NNNN}"""

def assemble_record(image_path: str, artefact_meta: dict,
                    transcription: dict, translation: dict,
                    processing_log: list) -> dict:
    """Assemble full record dict from all pipeline outputs."""

def save_record(record: dict, output_dir: str = "data/records") -> str:
    """Save record as JSON. Returns saved file path."""

def export_pdf(record: dict, output_dir: str = "outputs/exports") -> str:
    """Generate researcher-friendly PDF with side-by-side image comparison,
    transcription, translation, metadata table, and citation block."""
```

---

### Stage 7 — Main Pipeline (`src/pipeline.py`)

**Purpose:** Orchestrate all stages end-to-end for a single image or batch.

```python
def process_single(image_path: str, artefact_meta: dict,
                   script: str = "auto") -> dict:
    """
    Run full pipeline on one image.
    Returns completed record dict.
    """
    # 1. Preprocess
    preprocessed = preprocess(image_path, ...)
    # 2. Enhance
    enhanced = enhance(preprocessed, ...)
    # 3. Binarise
    binary = binarise(enhanced, ...)
    # 4. OCR
    transcription = transcribe(binary, script=script)
    # 5. Translate
    translation = translate(transcription, artefact_context=artefact_meta)
    # 6. Assemble record
    record = assemble_record(image_path, artefact_meta, transcription, translation, log)
    # 7. Save
    save_record(record)
    return record


def process_batch(input_dir: str, meta_csv: str = None,
                  workers: int = 4) -> list:
    """
    Process all images in input_dir.
    If meta_csv provided, read artefact metadata from CSV per image.
    Uses multiprocessing for parallel enhancement.
    """
```

**Batch metadata CSV format (`data/batch_meta.csv`):**

```csv
filename,type,material,period,location,script,condition
IMG_001.jpg,stone_inscription,granite,7th century CE,Mamallapuram,tamil,partial_damage
IMG_002.tif,palm_leaf,palm,12th century CE,Thanjavur,tamil,heavy_damage
IMG_003.jpg,copper_plate,copper,9th century CE,Kanchipuram,grantha,good
```

---

### Stage 8 — Gradio UI (`app.py`)

**Purpose:** Simple web interface for non-technical team members and researchers
to upload images, trigger processing, and browse results.

**Tabs to implement:**

1. **Process new image** — upload image, fill artefact metadata form, run pipeline, view record
2. **Browse records** — searchable list of all processed records with thumbnails
3. **View record** — detailed record card with before/after images, transcription
4. **Export** — download record as PDF or JSON
5. **Translation** *(disabled in Phase 1 — activated in Phase 2)* — view English translation when available

**Quick start:**

```python
import gradio as gr

def process_image(image, artefact_type, location, script, period):
    """Called by Gradio on form submit. Returns enhanced image + record summary."""
    meta = {"type": artefact_type, "location": location,
            "period": period, "script": script}
    record = process_single(image, meta, script=script)
    return record["images"]["enhanced"], record["transcription"]["text"], \
           record["translation"]["english"]

with gr.Blocks(title="Inscription Digitisation") as app:
    gr.Markdown("## Inscription & Manuscript Digitisation")
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(label="Upload inscription image")
            artefact_type = gr.Dropdown(
                ["stone_inscription", "palm_leaf", "copper_plate",
                 "paper_manuscript", "cave_painting"],
                label="Artefact type")
            script = gr.Dropdown(
                ["auto", "tamil", "sanskrit", "kannada",
                 "telugu", "malayalam", "brahmi", "grantha"],
                label="Script (select auto if unsure)")
            location = gr.Textbox(label="Location (site, district, state)")
            period = gr.Textbox(label="Estimated period (e.g. 7th century CE)")
            submit_btn = gr.Button("Process image")
        with gr.Column():
            enhanced_out = gr.Image(label="Enhanced image")
            transcription_out = gr.Textbox(label="Transcription")
            translation_out = gr.Textbox(label="English translation")
    submit_btn.click(process_image,
                     inputs=[img_input, artefact_type, location, script, period],
                     outputs=[enhanced_out, transcription_out, translation_out])

app.launch(share=False, server_port=7860)
```

---

## 6. Datasets

Use these datasets for development, testing, and model evaluation.

### Primary datasets (download first)

| Dataset | Contents | Access |
|---|---|---|
| Ancient Tamil Stone Inscriptions | Tamil stone inscriptions with LiDAR and 3D models | `kaggle.com/datasets/athiraishanmugam/ancient-tamil-stone-inscriptions` |
| Tamil Handwritten Palm Leaf (THPLMD) | 262 deteriorated Tamil palm leaf samples with binarised ground truth | ScienceDirect / PMC |
| Tamil Handwritten Character Corpus | Tamil handwritten characters across different centuries | `data.mendeley.com/datasets/6zcpgchvmx/1` |

### Secondary datasets

| Dataset | Contents | Access |
|---|---|---|
| Indiscapes (IIIT Hyderabad) | Layout annotations for historical Indic manuscripts | `ihdia.iiit.ac.in` |
| Sanskrit OCR Dataset | Classical Sanskrit document images | `github.com/ihdia/sanskrit-ocr` |
| Brahmi Character Dataset | 1032 Ashokan Brahmi characters, 258 classes | arxiv.org/abs/2501.01981 — contact authors |
| Kannada Inscriptions | Leaf manuscripts and stone inscriptions from Hampi | Contact Kuvempu Institute of Kannada Studies |

### Institutional archives (free to browse/download)

| Source | URL |
|---|---|
| Digital Library of India | `dli.ernet.in` |
| eGangotri | `egangotri.org` |
| IIIT Hyderabad IHDIA | `ihdia.iiit.ac.in` |
| Archaeological Survey of India | `asi.nic.in` |

### Download script

```bash
# Kaggle (requires kaggle CLI: pip install kaggle)
kaggle datasets download athiraishanmugam/ancient-tamil-stone-inscriptions
unzip ancient-tamil-stone-inscriptions.zip -d data/raw/tamil_stone/

# Mendeley (direct download)
# Visit data.mendeley.com/datasets/6zcpgchvmx/1 and download manually
# Extract to data/raw/tamil_handwritten/
```

---

## 7. Quality Evaluation

Every processed image must be evaluated. Log these metrics per image:

| Metric | What it measures | Tool |
|---|---|---|
| PSNR | Signal-to-noise ratio (higher = less noise) | `skimage.metrics.peak_signal_noise_ratio` |
| SSIM | Structural similarity to ground truth | `skimage.metrics.structural_similarity` |
| OCR confidence | Average word confidence from OCR engine | Built into Tesseract / EasyOCR output |
| Character error rate (CER) | OCR accuracy vs. ground truth text | `jiwer` library |

**Minimum acceptable thresholds:**
- PSNR >= 30 dB (for enhanced vs. original comparison)
- OCR confidence >= 0.70 average across image
- CER <= 0.15 on test set with known ground truth

```python
# Evaluation helper
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def evaluate_enhancement(original: np.ndarray,
                          enhanced: np.ndarray) -> dict:
    psnr = peak_signal_noise_ratio(original, enhanced)
    ssim = structural_similarity(original, enhanced, channel_axis=2)
    return {"psnr": round(psnr, 2), "ssim": round(ssim, 4)}
```

---

## 8. Non-Destructive Processing Rules

These rules are MANDATORY. Never break them.

1. **Never overwrite original files.** Raw images in `data/raw/` are read-only.
   All outputs go to `data/enhanced/`, `data/binarised/`, etc.

2. **Save processed images as JPEG (quality=95).** TIFF archival copies may be added in a future phase if storage requirements demand it.

3. **Follow the 3-2-1 backup rule:**
   - 3 copies of original data
   - 2 different storage media
   - 1 offsite / cloud backup

4. **Preserve EXIF metadata** through all processing stages using `piexif` or PIL.

5. **Log every operation** — what was done, when, by which pipeline version.

6. **Version control model weights** — record which version of each model was used
   in the processing log of every record.

---

## 9. Implementation Phases & Timeline

| Phase | What to build | Duration | Status |
|---|---|---|---|
| Phase 1 | Environment setup, folder structure, test on 5 sample images | Week 1 | ✅ In scope |
| Phase 2 | `preprocess.py` + `enhance.py` — full enhancement pipeline with batch processing | Weeks 2–3 | ✅ In scope |
| Phase 3 | `binarise.py` + `ocr.py` — binarisation and OCR for Tamil and Sanskrit first | Weeks 4–6 | ✅ In scope |
| Phase 4 | `translate.py` — translation layer with MT models + LLM fallback | Weeks 7–9 | 🔲 Phase 2 (time-permitting) |
| Phase 5 | `record.py` — record assembly, JSON storage, PDF export | Weeks 10–11 | ✅ In scope (translation field left blank) |
| Phase 6 | `app.py` — Gradio UI, browse/search interface | Weeks 12–14 | ✅ In scope (translation tab disabled) |

> **Mentor instruction (March 2026):** Deliver a robust, well-evaluated OCR and transcription
> system as the Phase 1 endpoint. Translation is Phase 2 — implement only if timeline permits
> after Phase 1 is complete and quality-evaluated.

**Start here on Day 1:**

```bash
# 1. Set up environment (see Section 4)
# 2. Download 5 test images from the Kaggle Tamil Stone dataset
# 3. Run this to verify the enhancement stage works:

python src/enhance.py --input data/raw/test_001.jpg \
                      --output data/enhanced/test_001_enhanced.tif \
                      --model models/weights/RealESRGAN_x4plus.pth
```

---

## 10. Key Decisions & Rationale

| Decision | Reason |
|---|---|
| Real-ESRGAN over basic upscaling | Trained on real-world degraded photos; preserves texture better than bicubic |
| DStretch for rock/cave art | Originally developed for rock art analysis; reveals colour invisible to human eye |
| Sauvola binarisation over Otsu | Handles uneven backgrounds (stone surface, aged palm leaf) better than global threshold |
| EasyOCR + Tesseract ensemble | Neither engine is perfect for ancient scripts; combining raises confidence |
| JSON record format | Human-readable, version-controllable, easy to export to any format |
| Gradio for UI | Zero frontend code needed; instant web interface; easily shareable |
| JPEG for preprocessed output | Avoids PIL/libtiff metadata write failures on Windows; smaller files; sufficient quality at 95 for subsequent pipeline stages |

---

## 11. Known Limitations & Future Work

- **Translation deferred to Phase 2:** As advised by the project mentor, translation is out of scope
  for the current phase. The architecture is fully designed (see Stage 5 above). Implementation
  will proceed in Phase 2 if the timeline permits after Phase 1 is complete and quality-evaluated.
  Standard MT models (Helsinki-NLP OPUS-MT) will be used for post-10th century CE texts;
  LLM fallback (Claude / GPT-4) for archaic classical forms.

- **Brahmi and Grantha scripts:** No off-the-shelf OCR model exists. These will require
  a custom trained model using the Brahmi character dataset. Flag these for manual transcription
  in the interim.

- **Very heavy damage:** Images with >50% surface damage will produce low-confidence OCR.
  Consider implementing an inpainting step using `LaMa` (Large Mask Inpainting) to reconstruct
  damaged regions before OCR.

- **3D inscriptions:** Deep-carved stone inscriptions benefit from 3D/LiDAR imaging (the Kaggle
  dataset includes 3D models). Future work: process 3D point cloud data as an alternative input.

- **Multi-script inscriptions:** Some inscriptions use two scripts (e.g. Tamil + Grantha).
  Future work: segment by script region before OCR.

- **Public portal:** Phase 7 (not yet planned) would expose processed records via a public-facing
  web portal built on Omeka S or a custom React frontend.

---

## 12. Interdisciplinary Team Roles

This is a three-branch interdisciplinary project. Each branch has a clearly defined ownership
area. No branch's work is optional — all three are required for the project to be complete.

> Note: The project works entirely on already-scanned existing images. There is no hardware
> acquisition component. Physical camera setup, lighting rigs, and field capture are out of scope.

---

### CS / IT — AI pipeline & software (core engine)

Owns the end-to-end software pipeline from image input to final record output.

| Responsibility | Files |
|---|---|
| Preprocessing pipeline | `src/preprocess.py` |
| AI enhancement (Real-ESRGAN, DStretch) | `src/enhance.py` |
| Binarisation | `src/binarise.py` |
| OCR & transcription | `src/ocr.py` |
| Translation layer | `src/translate.py` |
| Record assembly & PDF export | `src/record.py` |
| Pipeline orchestration | `src/pipeline.py` |
| Gradio web UI | `app.py` |

---

### ECE — Signal processing & image quality analysis

Owns the analytical and algorithmic depth of the image processing stages. The ECE student
does not build hardware — they bring rigorous signal processing theory to improve and
validate the image processing pipeline.

**1. Noise characterisation & modelling**
Analyse the types of noise present in existing scanned inscription images:
- Gaussian noise (scanner sensor noise)
- Salt-and-pepper noise (dust, damage)
- JPEG compression artefacts (blocking, ringing)
- Periodic noise (scanner line artifacts)

For each artefact type (stone, palm leaf, copper, paper, cave) produce a noise profile
that justifies which denoising algorithm and parameter settings should be used.
Deliverable: `docs/noise_analysis_report.pdf`

**2. Custom filter design**
Design and implement spatial and frequency-domain filters tuned specifically for
inscription images — going beyond the default OpenCV options:
- Gabor filter bank for separating inscription texture from background texture
- Directional edge enhancement filters for carving edge detection
- FFT-based periodic noise removal for scanner line artefacts

Implement in `src/filters.py`. These are called from `src/enhance.py` as optional
pre-processing steps selectable per artefact type.

```python
# src/filters.py — ECE-owned module
def gabor_filter_bank(img: np.ndarray,
                      frequencies: list = [0.1, 0.2, 0.4],
                      orientations: int = 8) -> np.ndarray:
    """Apply Gabor filter bank to separate text texture from background."""

def directional_edge_enhance(img: np.ndarray,
                              angle_deg: float = 45.0) -> np.ndarray:
    """Enhance edges in a specific direction — useful for carved inscriptions."""

def remove_periodic_noise_fft(img: np.ndarray,
                               threshold: float = 0.1) -> np.ndarray:
    """Use FFT to detect and remove periodic noise (scanner line artifacts)."""
```

**3. Histogram & colour channel analysis**
Perform per-channel statistical analysis of inscription images to:
- Understand the colour distribution differences across artefact types
- Tune DStretch colour space parameters (LAB vs YDS vs YBK) for each material
- Produce before/after histogram plots for the project report

```python
# src/analysis.py — ECE-owned module
def analyse_colour_distribution(img_path: str) -> dict:
    """Compute per-channel mean, std, skewness, kurtosis. Return as dict."""

def plot_histogram_comparison(original: np.ndarray,
                               enhanced: np.ndarray,
                               output_path: str) -> None:
    """Save side-by-side RGB histogram plot for before/after comparison."""
```

**4. Image quality metrics — own the evaluation layer**
Implement and interpret all quantitative quality metrics. This is the ECE student's
primary contribution to Section 7 (Quality Evaluation):
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- CNR (Contrast-to-Noise Ratio) — particularly useful for low-contrast inscriptions
- Edge sharpness score using Laplacian variance

```python
# src/metrics.py — ECE-owned module
def compute_psnr(original: np.ndarray, enhanced: np.ndarray) -> float:
    """Peak signal-to-noise ratio. Higher = better. Target >= 30 dB."""

def compute_ssim(original: np.ndarray, enhanced: np.ndarray) -> float:
    """Structural similarity. Range 0–1. Target >= 0.85."""

def compute_cnr(img: np.ndarray, text_mask: np.ndarray) -> float:
    """Contrast-to-noise ratio between text region and background."""

def compute_sharpness(img: np.ndarray) -> float:
    """Laplacian variance as proxy for edge sharpness. Higher = sharper."""

def full_quality_report(original: np.ndarray,
                         enhanced: np.ndarray,
                         text_mask: np.ndarray = None) -> dict:
    """Run all metrics and return consolidated quality report dict."""
```

Deliverable: integrate `full_quality_report()` into `src/pipeline.py` so every
processed image gets a quality score logged to its record.

---

### IEM — Process design, project management & impact analysis

Owns the operational, organisational, and business layer of the project. The IEM student
ensures the project is well-planned, efficiently run, and impactful beyond the codebase.

**1. Project management**
- Own and maintain the project Gantt chart (use MS Project, Excel, or Notion)
- Run weekly sprint planning and track task completion across the team
- Maintain a risk register — identify risks (e.g. OCR accuracy too low, dataset not available)
  and define mitigation strategies
- Deliverable: `docs/project_plan.xlsx` updated weekly

**2. Digitisation workflow design (lean process engineering)**
Apply industrial engineering principles to the inscription processing workflow:
- Map the current pipeline as a value stream map (VSM)
- Identify bottlenecks (e.g. the OCR stage takes 3x longer than enhancement)
- Propose and test process improvements (parallelisation, batching strategy, queue management)
- Measure throughput: how many inscriptions can be processed per hour?

Deliverable: `docs/workflow_analysis.pdf` with VSM diagram and throughput measurements.

**3. Cost-benefit & scalability analysis**
Answer the question: what does it cost to digitise 10,000 inscriptions, and what is
the value created?
- Estimate compute cost per inscription (cloud GPU vs local CPU)
- Estimate storage cost per inscription (TIFF master + access copies + records)
- Estimate researcher time saved vs. manual transcription
- Model scalability: what changes if volume goes from 100 → 10,000 → 1,000,000 records?

Deliverable: `docs/cost_benefit_analysis.pdf`

**4. Stakeholder documentation**
Write the documents that non-technical stakeholders (museum curators, archaeologists,
government bodies, ASI) need to understand and adopt the project:
- Project overview one-pager (non-technical)
- User guide for the Gradio UI
- Data governance policy (who owns the records, access control, licensing)
- Impact statement: why does this project matter?

Deliverables: `docs/user_guide.pdf`, `docs/impact_statement.pdf`

---

### Ownership summary

| Component | Owner |
|---|---|
| AI enhancement pipeline | CS/IT |
| OCR & transcription | CS/IT |
| Translation layer | CS/IT |
| Web UI / Gradio portal | CS/IT |
| Noise modelling & characterisation | ECE |
| Custom filter design (Gabor, FFT) | ECE |
| Colour channel & histogram analysis | ECE |
| Image quality metrics (PSNR, SSIM, CNR) | ECE |
| Project schedule & risk register | IEM |
| Value stream mapping & throughput | IEM |
| Cost-benefit & scalability analysis | IEM |
| Stakeholder & impact documentation | IEM |

### New files added by ECE and IEM

```
inscription-digitisation/
├── src/
│   ├── filters.py       ← ECE: custom Gabor, directional, FFT filters
│   ├── analysis.py      ← ECE: colour distribution and histogram tools
│   └── metrics.py       ← ECE: PSNR, SSIM, CNR, sharpness evaluation
└── docs/
    ├── noise_analysis_report.pdf     ← ECE deliverable
    ├── project_plan.xlsx             ← IEM deliverable
    ├── workflow_analysis.pdf         ← IEM deliverable
    ├── cost_benefit_analysis.pdf     ← IEM deliverable
    ├── user_guide.pdf                ← IEM deliverable
    └── impact_statement.pdf         ← IEM deliverable
```

---

## 13. Contact & References

- DStretch algorithm: Jon Harman — `dstretch.com`
- Real-ESRGAN paper: Wang et al. 2021 — `arxiv.org/abs/2107.10833`
- Indiscapes dataset: IIIT Hyderabad — `ihdia.iiit.ac.in`
- Brahmi OCR research: arxiv.org/abs/2501.01981
- Tamil NLP resources: `ai4bharat.org`
- Indic OCR models: `bhashini.gov.in`
- Gabor filter theory: Daugman, J. (1985) — Uncertainty relation for resolution in space, spatial frequency, and orientation
- SSIM metric: Wang et al. (2004) — Image quality assessment: from error visibility to structural similarity
- Value stream mapping: Rother, M. & Shook, J. — Learning to See (Lean Enterprise Institute)
- Cost modelling reference: AWS pricing calculator — `calculator.aws`

---

*Last updated: March 2026. Maintained by the project team.*
*Scope updated March 2026: Phase 1 endpoint is OCR & Transcription. Translation is Phase 2 (time-permitting) per mentor guidance.*
*Agent: read this file top-to-bottom before taking any action on this codebase.*
