import cv2
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from src.binarise import binarise, detect_document_type, _to_gray, detect_rubbing, binarise_stone, binarise_sauvola

def main():
    img_path = r"C:\6th semester EL's\Interdisciplinary project\Implementation\manuscript-digitisation\data\raw\Vijay Kumar extra images\img334.jpg"
    
    # Load image
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("Failed to load image!")
        return

    print("=== Image Stats ===")
    gray = _to_gray(img)
    print("Shape:", img.shape)
    print("Doc type:", detect_document_type(img))
    print("Gray mean:", gray.mean())
    print("Gray std:", gray.std())
    print("Is rubbing:", detect_rubbing(img))
    
    # Run binarise_stone directly
    stone_bin = binarise_stone(img)
    print("binarise_stone output mean:", stone_bin.mean())
    
    # Run binarise_sauvola directly
    sauvola_bin = binarise_sauvola(img)
    print("binarise_sauvola output mean:", sauvola_bin.mean())

    # Run the full binarise dispatcher
    out_path = "tune_img334_out/actual_binarise_output.png"
    bin_out = binarise(img_path, out_path, method="sauvola")
    print("Full binarise output mean:", bin_out.mean())
    
    # Save the direct stone and sauvola outputs
    cv2.imwrite("tune_img334_out/stone_bin.png", stone_bin)
    cv2.imwrite("tune_img334_out/sauvola_bin.png", sauvola_bin)

if __name__ == "__main__":
    main()
