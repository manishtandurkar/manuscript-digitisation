import cv2
import pytesseract

img_a = cv2.imread("scratch/test_method_a_inv.png", cv2.IMREAD_GRAYSCALE)
img_b = cv2.imread("scratch/segmental_clean_output.png", cv2.IMREAD_GRAYSCALE)

# Run OCR using Tamil language pack
config = "--oem 1 --psm 6"

def run_ocr(name, img):
    try:
        # Tesseract expects black text on white background (which both images have now)
        data = pytesseract.image_to_data(img, lang="tam", config=config, output_type=pytesseract.Output.DICT)
        text_list = [w for w in data['text'] if w.strip()]
        text = " ".join(text_list)
        confidences = [int(c) for c in data['conf'] if c != '-1']
        mean_conf = np.mean(confidences) if confidences else 0
        print(f"{name}: Mean Confidence = {mean_conf:.2f}% | Words detected = {len(text_list)} | Text: '{text}'")
    except Exception as e:
        print(f"Error running OCR on {name}: {e}")

import numpy as np
run_ocr("Method A (binarise_stone inverted)", img_a)
run_ocr("Refined Segmental Method", img_b)
