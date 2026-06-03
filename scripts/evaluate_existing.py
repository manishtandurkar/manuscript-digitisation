import logging
from pathlib import Path
import cv2
import numpy as np

from src.metrics import full_quality_report, compute_cnr

logging.basicConfig(level=logging.WARNING)

RAW_DIR = Path("data/raw")
VALIDATION_DIR = Path("data/validation_outputs")

TEST_IMAGES = [
    {
        "lang": "kannada",
        "raw_path": RAW_DIR / "kannada_stone" / "image2.jpeg",
        "val_folder": VALIDATION_DIR / "kannada" / "image2",
        "stem": "image2"
    },
    {
        "lang": "malayalam",
        "raw_path": RAW_DIR / "malayalam_stone" / "image1.jpeg",
        "val_folder": VALIDATION_DIR / "malayalam" / "image1",
        "stem": "image1"
    },
    {
        "lang": "tamil",
        "raw_path": RAW_DIR / "tamil_stone" / "tamil_001.jpg",
        "val_folder": VALIDATION_DIR / "tamil" / "tamil_001",
        "stem": "tamil_001"
    },
    {
        "lang": "telugu",
        "raw_path": RAW_DIR / "telugu_stone" / "image2.jpg",
        "val_folder": VALIDATION_DIR / "telugu" / "image2",
        "stem": "image2"
    },
    {
        "lang": "tulu",
        "raw_path": RAW_DIR / "tulu_stone" / "image5.png",
        "val_folder": VALIDATION_DIR / "tulu" / "image5",
        "stem": "image5"
    },
]

def main():
    print("Evaluating pre-existing binarisation outputs...\n")
    results = []

    for item in TEST_IMAGES:
        lang = item["lang"]
        raw_path = item["raw_path"]
        val_folder = item["val_folder"]
        stem = item["stem"]

        if not raw_path.exists():
            print(f"Warning: Raw image for {lang} not found at {raw_path}")
            continue
        if not val_folder.exists():
            print(f"Warning: Validation folder for {lang} not found at {val_folder}")
            continue

        raw_img = cv2.imread(str(raw_path))

        # We can evaluate different enhancement modes: preprocessed (which acts as base), dstretch, superres
        enh_modes = ["preprocessed", "dstretch", "superres"]
        bin_methods = ["sauvola", "otsu", "adaptive"]

        for mode in enh_modes:
            # Load enhanced image
            if mode == "preprocessed":
                enh_path = val_folder / f"{stem}_preprocessed.jpg"
            else:
                enh_path = val_folder / f"{stem}_enhanced_{mode}.jpg"

            if not enh_path.exists():
                continue

            enh_img = cv2.imread(str(enh_path))
            if enh_img is None:
                continue

            for method in bin_methods:
                bin_path = val_folder / f"{stem}_{mode}_binarised_{method}.png"
                if not bin_path.exists():
                    continue

                bin_img = cv2.imread(str(bin_path), cv2.IMREAD_GRAYSCALE)
                if bin_img is None:
                    continue

                try:
                    # Compute quality metrics using binarised image as mask
                    report = full_quality_report(raw_img, enh_img, text_mask=bin_img)
                    cnr_val = compute_cnr(enh_img, bin_img)

                    results.append({
                        "Language": lang,
                        "Image": raw_path.name,
                        "EnhMode": mode,
                        "BinMethod": method,
                        "PSNR": report["psnr"],
                        "SSIM": report["ssim"],
                        "CNR": cnr_val,
                        "PassesAll": report["passes_thresholds"]
                    })
                except Exception as e:
                    print(f"Failed metrics on {lang} / {mode} / {method}: {e}")

    # Print results table
    print("--- Pipeline Evaluation Results (Original to Binarisation) ---")
    header = f"{'Language':<10} | {'EnhMode':<12} | {'BinMethod':<10} | {'PSNR (dB)':<9} | {'SSIM':<6} | {'CNR':<6} | {'Passes':<6}"
    print(header)
    print("-" * len(header))

    # Sort results by language and mode
    results.sort(key=lambda x: (x["Language"], x["EnhMode"], x["BinMethod"]))
    
    for r in results:
        cnr_str = f"{r['CNR']:.2f}" if r['CNR'] != float('inf') else "inf"
        passes_str = "Yes" if r['PassesAll'] else "No"
        print(f"{r['Language']:<10} | {r['EnhMode']:<12} | {r['BinMethod']:<10} | {r['PSNR']:<9.2f} | {r['SSIM']:<6.4f} | {cnr_str:<6} | {passes_str:<6}")

    print("\n--- Worst Performing Analysis ---")
    if results:
        # Worst SSIM
        worst_ssim = min(results, key=lambda x: x["SSIM"])
        print(f"Lowest SSIM (Less structural similarity):")
        print(f"  * {worst_ssim['Language']} / {worst_ssim['EnhMode']} / {worst_ssim['BinMethod']} -> SSIM = {worst_ssim['SSIM']:.4f} (PSNR = {worst_ssim['PSNR']:.2f} dB, CNR = {worst_ssim['CNR']:.2f})")

        # Worst PSNR
        worst_psnr = min(results, key=lambda x: x["PSNR"])
        print(f"Lowest PSNR (Most noise/distortion added):")
        print(f"  * {worst_psnr['Language']} / {worst_psnr['EnhMode']} / {worst_psnr['BinMethod']} -> PSNR = {worst_psnr['PSNR']:.2f} dB (SSIM = {worst_psnr['SSIM']:.4f}, CNR = {worst_psnr['CNR']:.2f})")

        # Worst CNR
        valid_cnr = [x for x in results if x["CNR"] != float("inf") and not np.isnan(x["CNR"])]
        if valid_cnr:
            worst_cnr = min(valid_cnr, key=lambda x: x["CNR"])
            print(f"Lowest CNR (Worst text/background separation):")
            print(f"  * {worst_cnr['Language']} / {worst_cnr['EnhMode']} / {worst_cnr['BinMethod']} -> CNR = {worst_cnr['CNR']:.2f} (PSNR = {worst_cnr['PSNR']:.2f} dB, SSIM = {worst_cnr['SSIM']:.4f})")

if __name__ == "__main__":
    main()
