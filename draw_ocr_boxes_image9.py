#!/usr/bin/env python3
"""
Script to visualize how OCR is applied to image9.png.
Generates an annotated image showing the word bounding boxes detected by Tesseract.
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

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

def draw_ocr_visualization():
    binarised_path = project_root / "data" / "binarised" / "malayalam_stone__image9_png_binarised.png"
    annotated_output_path = project_root / "data" / "binarised" / "malayalam_stone__image9_png_ocr_boxes.png"

    print("==================================================")
    print("Generating OCR Bounding Box Visualization")
    print("==================================================")
    
    if not binarised_path.exists():
        print(f"ERROR: Binarised image not found at {binarised_path}")
        return

    # Load binarised image
    bin_img = cv2.imread(str(binarised_path))
    if bin_img is None:
        print(f"ERROR: Could not load binarised image")
        return

    # Create a copy to draw on (convert to color so we can draw green boxes)
    annotated_img = bin_img.copy()
    if annotated_img.ndim == 2 or annotated_img.shape[2] == 1:
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_GRAY2BGR)

    try:
        import pytesseract
    except ImportError:
        print("ERROR: pytesseract is not installed.")
        return

    tess_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if not Path(tess_cmd).exists():
        print(f"ERROR: Tesseract binary not found at {tess_cmd}")
        return

    pytesseract.pytesseract.tesseract_cmd = tess_cmd
    os.environ["TESSDATA_PREFIX"] = "tessdata"

    print("Running Tesseract to retrieve word coordinates...")
    try:
        custom_config = '--oem 1 --psm 6'
        data = pytesseract.image_to_data(bin_img, lang="mal", config=custom_config, output_type=pytesseract.Output.DICT)
    except Exception as exc:
        print(f"ERROR: Tesseract run failed: {exc}")
        return

    # Draw bounding boxes
    n_boxes = len(data['level'])
    box_count = 0
    print(f"Detected {n_boxes} layout elements. Extracting valid word boxes...")

    for i in range(n_boxes):
        conf = float(data['conf'][i])
        text = data['text'][i].strip()
        
        # Only draw boxes for actual words with positive confidence
        if conf > 0 and text:
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            
            # Draw rectangle (Green, thickness=2)
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 220, 0), 2)
            
            # Put label background
            cv2.rectangle(annotated_img, (x, y - 15), (x + int(w * 0.8), y), (0, 220, 0), -1)
            
            # Put confidence/text label (White, scale=0.35)
            label = f"{conf:.0f}%"
            cv2.putText(annotated_img, label, (x + 2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
            
            box_count += 1
            print(f"  Word: '{text}' | Box: [{x}, {y}, {w}, {h}] | Conf: {conf}%")

    # Save output
    cv2.imwrite(str(annotated_output_path), annotated_img)
    print("==================================================")
    print(f"Successfully drew {box_count} OCR bounding boxes.")
    print(f"Before Image: {binarised_path}")
    print(f"After Image:  {annotated_output_path}")
    print("==================================================")

if __name__ == "__main__":
    draw_ocr_visualization()
