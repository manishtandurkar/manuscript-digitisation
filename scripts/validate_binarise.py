import logging
from pathlib import Path
import cv2
import numpy as np

# Set up logging to console
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
LOGGER = logging.getLogger("validate_binarise")

from src.preprocess import preprocess
from src.enhance import enhance
from src.binarise import binarise, detect_document_type

# Paths
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/validation_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Select test images
TEST_IMAGES = [
    {"lang": "kannada", "path": RAW_DIR / "kannada_stone" / "image2.jpeg"},
    {"lang": "malayalam", "path": RAW_DIR / "malayalam_stone" / "image1.jpeg"},
    {"lang": "tamil", "path": RAW_DIR / "tamil_stone" / "tamil_001.jpg"},
    {"lang": "telugu", "path": RAW_DIR / "telugu_stone" / "image2.jpg"},
    {"lang": "tulu", "path": RAW_DIR / "tulu_stone" / "image5.png"},
]

def main():
    LOGGER.info("Starting binarisation validation script...")
    
    for item in TEST_IMAGES:
        lang = item["lang"]
        img_path = item["path"]
        
        if not img_path.exists():
            LOGGER.warning("Image for %s not found at %s", lang, img_path)
            continue
            
        LOGGER.info("=========================================")
        LOGGER.info("Processing %s: %s", lang, img_path.name)
        
        # Create output directories for this image
        img_out_dir = OUT_DIR / lang / img_path.stem
        img_out_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Preprocess
        pre_out = img_out_dir / f"{img_path.stem}_preprocessed.jpg"
        try:
            preprocess(str(img_path), str(pre_out))
            LOGGER.info("Step 1: Preprocessing succeeded.")
        except Exception as exc:
            LOGGER.error("Step 1: Preprocessing failed: %s", exc)
            continue
            
        # 2. Enhance - DStretch
        enh_dstretch_out = img_out_dir / f"{img_path.stem}_enhanced_dstretch.jpg"
        try:
            enhance(str(pre_out), str(enh_dstretch_out), mode="dstretch")
            LOGGER.info("Step 2a: Enhancement (DStretch) succeeded.")
        except Exception as exc:
            LOGGER.error("Step 2a: Enhancement (DStretch) failed: %s", exc)
            
        # 2b. Enhance - RealESRGAN
        enh_superres_out = img_out_dir / f"{img_path.stem}_enhanced_superres.jpg"
        try:
            # We run it; if Real-ESRGAN is missing it falls back to sharpening
            enhance(str(pre_out), str(enh_superres_out), mode="superres")
            LOGGER.info("Step 2b: Enhancement (Super-resolution) succeeded.")
        except Exception as exc:
            LOGGER.error("Step 2b: Enhancement (Super-resolution) failed: %s", exc)
            
        # 3. Detect document type on preprocessed image
        pre_img = cv2.imread(str(pre_out))
        if pre_img is not None:
            doc_type = detect_document_type(pre_img, img_path=pre_out)
            LOGGER.info("Detected document type: %s", doc_type)
        else:
            doc_type = "unknown"
            
        # 4. Binarise
        # We will test binarisation on preprocessed, dstretch-enhanced, and superres-enhanced images
        inputs = [
            ("preprocessed", pre_out),
            ("dstretch", enh_dstretch_out),
            ("superres", enh_superres_out),
        ]
        
        methods = ["sauvola", "otsu", "adaptive"]
        
        for input_name, input_path in inputs:
            if not input_path.exists():
                continue
                
            for method in methods:
                bin_out = img_out_dir / f"{img_path.stem}_{input_name}_binarised_{method}.png"
                try:
                    binarise(str(input_path), str(bin_out), method=method)
                    
                    # Verify result
                    bin_img = cv2.imread(str(bin_out), cv2.IMREAD_GRAYSCALE)
                    if bin_img is not None:
                        unique_vals = np.unique(bin_img)
                        is_bin = set(unique_vals).issubset({0, 255})
                        fg_percent = (cv2.countNonZero(bin_img) / bin_img.size) * 100
                        LOGGER.info(
                            "Binarise [%s - %s]: Succeeded. Is binary: %s, FG pixel%%: %.1f%%",
                            input_name, method, is_bin, fg_percent
                        )
                    else:
                        LOGGER.error("Binarise [%s - %s]: Output is not readable", input_name, method)
                except Exception as exc:
                    LOGGER.error("Binarise [%s - %s] failed: %s", input_name, method, exc)

    LOGGER.info("Validation complete. Output saved under: %s", OUT_DIR)

if __name__ == "__main__":
    main()
