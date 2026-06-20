# Teammate Reference: Digitisation Project Handover

> **Welcome!** This file is your one-shot reference guide to completing and submitting the Interdisciplinary Inscription & Manuscript Digitisation Project tomorrow.
> 
> To save you time, the core binarisation masking, custom ECE filter engines, color statistical analyses, and OCR automation are **already implemented, tested, and ready**. You only need to run, verify, and document the final outputs.

---

## 1. Project Folder & Status

* **Raw Scans (Read-Only):** `data/raw/`
* **Processed Outputs:** `data/enhanced/`, `data/binarised/`, `data/transcriptions/`, and `data/records/`
* **Automated Batch Processor:** [process_representative_samples.py](file:///c:/6th%20semester%20EL's/Interdisciplinary%20project/Implementation/manuscript-digitisation/scripts/process_representative_samples.py)
* **Custom ECE Spatial/Frequency Filters:** [filters.py](file:///c:/6th%20semester%20EL's/Interdisciplinary%20project/Implementation/manuscript-digitisation/src/filters.py)
* **Custom ECE Color/Histogram Analyzers:** [analysis.py](file:///c:/6th%20semester%20EL's/Interdisciplinary%20project/Implementation/manuscript-digitisation/src/analysis.py)

---

## 2. Core Tasks by Role

### 💻 CS / IT Student (Software Engine & Web UI)
1. **Start the Servers:**
   Ensure both the React frontend and FastAPI backend run concurrently:
   ```bash
   # Terminal 1: Backend API (port 8000)
   conda activate inscriptions
   uvicorn api.main:app --reload --port 8000

   # Terminal 2: React Frontend (port 5173)
   cd web
   npm run dev
   ```
2. **Web Portal Features:**
   * Open `http://localhost:5173` in your browser.
   * Verify the sidebar navigation, job list polling, and the **Before/After slider** for comparing original vs enhanced images.
3. **Verify OCR Engine:**
   Run the unit tests to make sure Tesseract and EasyOCR are working:
   ```bash
   pytest tests/test_ocr.py
   pytest tests/test_api.py
   ```

---

### 📡 ECE Student (Signal Processing & Quality Analytics)
All core custom signal processing filters and analytical modules have been created:
1. **Custom Filter Bank ([filters.py](file:///c:/6th%20semester%20EL's/Interdisciplinary%20project/Implementation/manuscript-digitisation/src/filters.py)):**
   * `gabor_filter_bank()`: Isolates character strokes from background noise in the spatial domain.
   * `directional_edge_enhance()`: Sobel-gradient-based projection filters for low-contrast carvings.
   * `remove_periodic_noise_fft()`: Uses 2D Fast Fourier Transform to zero out periodic/horizontal scanning lines in the frequency domain.
2. **Channel Distribution Analyzer ([analysis.py](file:///c:/6th%20semester%20EL's/Interdisciplinary%20project/Implementation/manuscript-digitisation/src/analysis.py)):**
   * `analyse_colour_distribution()`: Calculates per-channel statistical skewness, kurtosis, standard deviation, and mean.
   * `plot_histogram_comparison()`: Automatically outputs side-by-side RGB histogram comparisons.
3. **ECE Deliverable checklist:**
   Ensure `docs/noise_analysis_report.pdf` is populated with definitions of scanner sensor noise (Gaussian), paper grain (speckle), and periodic lines (frequency spikes).

---

### 📊 IEM Student (Process Management & Operations)
1. **Gantt Chart & Risks:** Update `docs/project_plan.xlsx` to show 100% completion of Phase 1 (through Stage 4).
2. **Throughput VSM:**
   Verify average throughput. The pipeline takes $\sim 1.5$ seconds for preprocessing, $\sim 12.0$ seconds for super-resolution (Real-ESRGAN CPU fallback), and $\sim 3.0$ seconds for Tesseract/EasyOCR ensemble transcription.
3. **Cost-Benefit PDF:** Write the cost breakdown for hosting the portal on standard AWS EC2/S3 vs processing on local hardware.
4. **Public User Manual:** Provide `docs/user_guide.pdf` showing screenshots of the React compare slider.

---

## 3. Running Binarisation & OCR on Representative Originals

A dedicated runner script is implemented at [process_representative_samples.py](file:///c:/6th%20semester%20EL's/Interdisciplinary project/Implementation/manuscript-digitisation/scripts/process_representative_samples.py).

This runner recursively searches for original representative scans (files ending with `_original.*` under `data/binarised_representative_samples/`) and processes them:

### Commands to Run:
```bash
# Run with default Sauvola local thresholding and auto script-detect
python scripts/process_representative_samples.py
```

### The "Text Exactly" Representation Requirement:
The script implements **Text Masking** to isolate text strokes while blacking out the background:
* **Option 1 (Original Color Masking):** Background is set to black, while the text pixels retain their exact original colors, textures, and carving details. Saved automatically as `*_masked.png`.
* **Option 2 (Pure Binary Polarity):** Background is black, text is pure white. Saved automatically as `*_binarised.png`.
* **Command Option:** If your mentor prefers `_binarised.png` to directly refer to the masked color image, run:
  ```bash
  python scripts/process_representative_samples.py --overwrite-binarised
  ```

---

## 4. Submission Checklist

Verify that the following files exist and are finalized before packaging:

| Stage / Role | Deliverable File | Status / Verification Command |
|---|---|---|
| **CS Core** | `scripts/process_representative_samples.py` | `python scripts/process_representative_samples.py` |
| **CS Core** | `src/preprocess.py`, `src/enhance.py`, `src/binarise.py`, `src/ocr.py`, `src/record.py` | `pytest tests/` |
| **ECE** | `src/filters.py` (Gabor, Sobel, FFT) | Imported inside `src/enhance.py` |
| **ECE** | `src/analysis.py` (RGB stats, Skewness, Kurtosis) | `python -c "import src.analysis as a; print(a.analyse_colour_distribution('data/binarised_representative_samples/kannada_stone/image3_original.jpeg'))"` |
| **ECE** | `docs/noise_analysis_report.pdf` | Ensure PDF contains noise profiles |
| **IEM** | `docs/project_plan.xlsx` | Gantt chart & Risk register |
| **IEM** | `docs/workflow_analysis.pdf` | Value stream map (VSM) & Bottlenecks |
| **IEM** | `docs/cost_benefit_analysis.pdf` | CPU vs GPU cost comparison |
| **IEM** | `docs/user_guide.pdf` | Web UI navigation screenshots |
| **IEM** | `docs/impact_statement.pdf` | Archaeological & historical value statement |
