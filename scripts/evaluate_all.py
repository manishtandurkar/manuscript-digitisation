import logging
from pathlib import Path
import cv2
import numpy as np

from src.preprocess import preprocess
from src.enhance import enhance
from src.binarise import binarise, detect_document_type
from src.metrics import full_quality_report, compute_cnr

logging.basicConfig(level=logging.WARNING)
LOGGER = logging.getLogger("evaluate_all")

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/evaluation_metrics_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_IMAGES = [
    {"lang": "kannada", "path": RAW_DIR / "kannada_stone" / "image2.jpeg"},
    {"lang": "malayalam", "path": RAW_DIR / "malayalam_stone" / "image1.jpeg"},
    {"lang": "tamil", "path": RAW_DIR / "tamil_stone" / "tamil_001.jpg"},
    {"lang": "telugu", "path": RAW_DIR / "telugu_stone" / "image2.jpg"},
    {"lang": "tulu", "path": RAW_DIR / "tulu_stone" / "image5.png"},
]

def main():
    print("Evaluating pipeline performance (Original -> Preprocess -> Enhance -> Binarise)...")
    results = []

    for item in TEST_IMAGES:
        lang = item["lang"]
        img_path = item["path"]

        if not img_path.exists():
            print(f"Warning: Image for {lang} not found at {img_path}")
            continue

        print(f"Processing {lang}: {img_path.name}...")

        # Load and resize if too large
        orig_img = cv2.imread(str(img_path))
        if orig_img is None:
            print(f"Failed to read raw image: {img_path}")
            continue
        h, w = orig_img.shape[:2]
        max_dim = 800
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            orig_img = cv2.resize(orig_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        
        resized_raw_path = OUT_DIR / f"{img_path.stem}_resized_raw.jpg"
        cv2.imwrite(str(resized_raw_path), orig_img)

        # 1. Preprocess
        pre_out = OUT_DIR / f"{img_path.stem}_preprocessed.jpg"
        preprocess(str(resized_raw_path), str(pre_out))

        # 2. Enhance
        enh_out = OUT_DIR / f"{img_path.stem}_enhanced.jpg"
        enhance(str(pre_out), str(enh_out), mode="superres")

        # 3. Read images
        # Reload resized original and enhanced image
        orig_img = cv2.imread(str(resized_raw_path))
        enh_img = cv2.imread(str(enh_out))
        doc_type = detect_document_type(orig_img)

        # 4. Binarise using sauvola, otsu, adaptive
        methods = ["sauvola", "otsu", "adaptive"]
        for method in methods:
            bin_out = OUT_DIR / f"{img_path.stem}_binarised_{method}.png"
            try:
                bin_img = binarise(str(enh_out), str(bin_out), method=method)
                
                # Compute metrics
                report = full_quality_report(orig_img, enh_img, text_mask=bin_img)
                
                # Also compute CNR directly using the binarised mask
                cnr_val = compute_cnr(enh_img, bin_img)
                
                results.append({
                    "Language": lang,
                    "Image": img_path.name,
                    "DocType": doc_type,
                    "Method": method,
                    "PSNR": report["psnr"],
                    "SSIM": report["ssim"],
                    "CNR": cnr_val,
                    "PassesAll": report["passes_thresholds"],
                    "PassesPSNR": report["threshold_check"]["psnr"],
                    "PassesSSIM": report["threshold_check"]["ssim"],
                    "PassesCNR": report["threshold_check"]["cnr"],
                })
            except Exception as e:
                print(f"Failed binarisation with method {method}: {e}")

    # Format output as a nice text table
    print("\n--- Pipeline Evaluation Results (Original to Binarisation) ---")
    header = f"{'Language':<10} | {'Image':<15} | {'DocType':<10} | {'Method':<10} | {'PSNR (dB)':<9} | {'SSIM':<6} | {'CNR':<6} | {'Passes':<6}"
    print(header)
    print("-" * len(header))
    for r in results:
        cnr_str = f"{r['CNR']:.2f}" if r['CNR'] != float('inf') else "inf"
        passes_str = "Yes" if r['PassesAll'] else "No"
        print(f"{r['Language']:<10} | {r['Image']:<15} | {r['DocType']:<10} | {r['Method']:<10} | {r['PSNR']:<9.2f} | {r['SSIM']:<6.4f} | {cnr_str:<6} | {passes_str:<6}")

    # Determine worst performing image
    print("\n--- Worst Performing Analysis ---")
    if results:
        # Worst SSIM
        worst_ssim = min(results, key=lambda x: x["SSIM"])
        print(f"Worst SSIM: {worst_ssim['Language']} / {worst_ssim['Image']} (Method: {worst_ssim['Method']}) -> SSIM = {worst_ssim['SSIM']:.4f}")

        # Worst PSNR
        worst_psnr = min(results, key=lambda x: x["PSNR"])
        print(f"Worst PSNR: {worst_psnr['Language']} / {worst_psnr['Image']} (Method: {worst_psnr['Method']}) -> PSNR = {worst_psnr['PSNR']:.2f} dB")

        # Worst CNR
        # filter out inf or handle it
        worst_cnr = min(results, key=lambda x: x["CNR"] if x["CNR"] != float("inf") else 999999.0)
        cnr_val_str = f"{worst_cnr['CNR']:.2f}" if worst_cnr['CNR'] != float('inf') else "inf"
        print(f"Worst CNR: {worst_cnr['Language']} / {worst_cnr['Image']} (Method: {worst_cnr['Method']}) -> CNR = {cnr_val_str}")

if __name__ == "__main__":
    main()

