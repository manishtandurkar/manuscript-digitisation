#!/usr/bin/env python3
"""
Automates binarisation, color text masking, and OCR for representative sample images.
Scans data/binarised_representative_samples/ for files ending with '_original.*' and:
1. Runs binarisation to produce a black-and-white mask.
2. Applies the mask to the original image to keep text in its original color while blacking out the background.
3. Runs the OCR ensemble to produce structured transcription JSONs.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure root workspace is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.binarise import binarise
    from src.ocr import transcribe
except ImportError:
    print("Error: Could not import src.binarise or src.ocr. Make sure you run from the project directory.")
    sys.exit(1)


def apply_text_mask(original_path: Path, binarised_mask_path: Path, output_path: Path) -> np.ndarray:
    """
    Masks the original image so that:
    - Whichever pixel is NOT text (i.e. black/0 in the mask) becomes black.
    - Whichever pixel IS text (i.e. white/255 in the mask) is represented EXACTLY as in the original image.
    """
    # Read the original image (in color)
    img = cv2.imdecode(np.fromfile(str(original_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read original image: {original_path}")

    # Read the binarised mask (grayscale)
    mask = cv2.imdecode(np.fromfile(str(binarised_mask_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read binarised mask: {binarised_mask_path}")

    # Resize mask to original image size if they mismatch (safety check)
    if img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Perform bitwise AND to mask the original image
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    # Save output
    success = cv2.imwrite(str(output_path), masked_img)
    if not success:
        raise IOError(f"Failed to write masked image to {output_path}")

    return masked_img


def process_all_samples(method: str = "sauvola", overwrite_binarised_with_mask: bool = False):
    samples_dir = PROJECT_ROOT / "data" / "binarised_representative_samples"
    if not samples_dir.exists():
        print(f"Error: Directory not found: {samples_dir}")
        return

    # Find all original images recursively
    original_images: list[Path] = []
    for ext in ("*.jpeg", "*.jpg", "*.JPG", "*.png"):
        original_images.extend(samples_dir.rglob(f"*_original{ext}"))

    original_images = sorted(list(set(original_images)))

    if not original_images:
        print("No representative images ending with '_original.*' found.")
        return

    print(f"Found {len(original_images)} representative original images to process.\n")

    for i, orig_path in enumerate(original_images, start=1):
        print(f"[{i}/{len(original_images)}] Processing: {orig_path.relative_to(PROJECT_ROOT)}")
        t0 = time.monotonic()

        # Output paths
        lang_dir = orig_path.parent
        stem = orig_path.name.split("_original")[0]
        
        # Standard binarised path
        bin_path = lang_dir / f"{stem}_binarised.png"
        
        # Masked text path (original text on black background)
        masked_path = lang_dir / f"{stem}_masked.png"
        
        # OCR transcription output
        ocr_path = lang_dir / f"{stem}_transcription.json"

        # 1. Run Binarisation
        try:
            print("  -> Binarising...")
            binarise(str(orig_path), str(bin_path), method=method)
            print(f"     Saved binary mask to {bin_path.name}")
        except Exception as exc:
            print(f"     [ERROR] Binarisation failed: {exc}")
            continue

        # 2. Run Color/Original Text Masking
        try:
            print("  -> Masking original text on black background...")
            apply_text_mask(orig_path, bin_path, masked_path)
            print(f"     Saved masked text image to {masked_path.name}")
            
            if overwrite_binarised_with_mask:
                # If requested, replace '_binarised.png' with the masked image
                import shutil
                shutil.copy2(masked_path, bin_path)
                print("     [Note] Overwrote '_binarised.png' with masked image as requested.")
        except Exception as exc:
            print(f"     [ERROR] Masking failed: {exc}")
            continue

        # 3. Run OCR / Transcription
        # We run OCR on the binarised mask image for standard processing, or the masked image
        ocr_input = bin_path
        try:
            print("  -> Running OCR...")
            script_name = "auto"
            # Deduce script based on directory name if possible
            dir_name = lang_dir.name.lower()
            for known_script in ("tamil", "sanskrit", "kannada", "telugu", "malayalam"):
                if known_script in dir_name:
                    script_name = known_script
                    break
            
            tx = transcribe(str(ocr_input), script=script_name, output_path=str(ocr_path))
            print(f"     OCR Finished. Detected Script: {tx['script']}, Confidence: {tx['overall_confidence']:.2f}")
            print(f"     Saved OCR JSON to {ocr_path.name}")
        except Exception as exc:
            print(f"     [ERROR] OCR failed: {exc}")

        duration = time.monotonic() - t0
        print(f"  Finished in {duration:.2f}s\n")


def main():
    parser = argparse.ArgumentParser(description="Process representative original samples")
    parser.add_argument(
        "--method",
        choices=["sauvola", "otsu", "adaptive", "unet", "docentr"],
        default="sauvola",
        help="Binarisation method to use (default: sauvola)",
    )
    parser.add_argument(
        "--overwrite-binarised",
        action="store_true",
        help="If set, overwrites the '_binarised.png' files with the masked original text images.",
    )
    args = parser.parse_args()

    process_all_samples(method=args.method, overwrite_binarised_with_mask=args.overwrite_binarised)


if __name__ == "__main__":
    main()
