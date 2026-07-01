#!/usr/bin/env python3
"""
Standalone script to binarise and transcribe image9.png (Malayalam Stone Inscription).
Outputs the Malayalam Unicode text and confidence statistics to the console and saves them to a file.
"""
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Reconfigure stdout to use UTF-8 so we can print Malayalam in Windows terminal logs
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to path to load core modules
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.preprocess import preprocess
from src.enhance import enhance
from src.binarise import binarise

def run_image9_pipeline():
    raw_img_path = project_root / "data" / "raw" / "malayalam_stone" / "image9.png"
    preprocessed_path = project_root / "data" / "preprocessed" / "malayalam_stone__image9_png_preprocessed.jpg"
    enhanced_path = project_root / "data" / "enhanced" / "malayalam_stone__image9_png_enhanced.jpg"
    binarised_path = project_root / "data" / "binarised" / "malayalam_stone__image9_png_binarised.png"
    output_text_path = project_root / "image9_ocr_output.txt"

    print("==================================================")
    print("STEP 1: Checking Raw Image")
    print("==================================================")
    if not raw_img_path.exists():
        print(f"ERROR: Raw image not found at {raw_img_path}")
        return
    print(f"Found raw image: {raw_img_path}")
    print(f"Size: {raw_img_path.stat().st_size} bytes")

    # Ensure output directories exist
    preprocessed_path.parent.mkdir(parents=True, exist_ok=True)
    enhanced_path.parent.mkdir(parents=True, exist_ok=True)
    binarised_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n==================================================")
    print("STEP 2: Running Pipeline (Preprocess -> Enhance -> Binarise)")
    print("==================================================")
    
    print("Running Preprocessing (CLAHE + White Balance)...")
    preprocess(str(raw_img_path), str(preprocessed_path))
    print(f"-> Preprocessed image saved to: {preprocessed_path}")

    print("Running Signal Enhancement (NLM Denoise + Sharpening)...")
    enhance(str(preprocessed_path), str(enhanced_path), use_dstretch=False)
    print(f"-> Enhanced image saved to: {enhanced_path}")

    print("Running Binarisation (Sauvola Stone Inscription Pipeline)...")
    binarise(str(enhanced_path), str(binarised_path), method="sauvola")
    print(f"-> Binarised image saved to: {binarised_path}")

    print("\n==================================================")
    print("STEP 3: Running OCR on Binarised Image (Tesseract)")
    print("==================================================")
    try:
        import pytesseract
    except ImportError:
        print("ERROR: pytesseract is not installed in the python environment.")
        return

    tess_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if not Path(tess_cmd).exists():
        print(f"ERROR: Tesseract binary not found at {tess_cmd}")
        return

    # Configure Pytesseract to use Windows Tesseract installation
    pytesseract.pytesseract.tesseract_cmd = tess_cmd
    
    # Configure TESSDATA_PREFIX to use local project tessdata containing mal.traineddata
    # We use a relative path "tessdata" to avoid single-quote space parsing issues on Windows
    os.environ["TESSDATA_PREFIX"] = "tessdata"
    
    # Load binarised image
    bin_img = cv2.imread(str(binarised_path))
    if bin_img is None:
        print(f"ERROR: Could not load binarised image from {binarised_path}")
        return

    print("Running Malayalam text extraction...")
    t0 = time.time()
    try:
        # Run Tesseract relying on TESSDATA_PREFIX environment variable
        custom_config = '--oem 1 --psm 6'
        ocr_text = pytesseract.image_to_string(bin_img, lang="mal", config=custom_config)
        duration = time.time() - t0
        print(f"OCR finished in {duration:.2f} seconds.")
    except Exception as exc:
        print(f"ERROR: OCR execution failed: {exc}")
        return

    print("\n==================================================")
    print("OCR RESULTS FOR IMAGE9.PNG")
    print("==================================================")
    print(ocr_text)
    print("==================================================")

    # Save to file
    with open(output_text_path, "w", encoding="utf-8") as fh:
        fh.write("=== OCR Results for malayalam_stone/image9.png ===\n")
        fh.write(f"Processed on: {time.asctime()}\n")
        fh.write("==================================================\n\n")
        fh.write(ocr_text)
    
    print(f"\nSaved full transcription output to: {output_text_path}")
    print("==================================================")

if __name__ == "__main__":
    run_image9_pipeline()
