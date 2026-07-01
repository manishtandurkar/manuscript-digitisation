import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from src.binarise import binarise

input_path = "data/raw/tamil_stone/IMG_3924.jpg"
output_path = "scratch/IMG_3924_binarised_main.png"

try:
    print(f"Calling binarise on {input_path}...")
    result = binarise(input_path, output_path)
    print("Success calling binarise!")
    
    out_p = Path(output_path)
    if out_p.exists():
        img = cv2.imread(str(out_p), cv2.IMREAD_GRAYSCALE)
        H, W = img.shape
        print(f"Output exists! Shape: {img.shape}")
        
        # Crop same region to compare with verification crop
        cy, cx = H // 2, W // 2
        crop_integrated = img[cy-400:cy+400, cx-400:cx+400]
        
        # Count black and white pixels (white text on black background)
        total = crop_integrated.size
        black = np.sum(crop_integrated == 0)
        white = np.sum(crop_integrated == 255)
        print(f"Crop stats: black={black} ({black/total*100:.2f}%), white={white} ({white/total*100:.2f}%)")
        
        # Check against Option C stand-alone (saved inverted, so we invert crop_integrated to compare)
        crop_integrated_inv = cv2.bitwise_not(crop_integrated)
        option_c = cv2.imread("scratch/img3924_crop_m31_ws101_k25_inv.png", cv2.IMREAD_GRAYSCALE)
        
        if option_c is not None:
            diff = cv2.absdiff(crop_integrated_inv, option_c)
            non_zero = np.count_nonzero(diff)
            print(f"Difference pixels between main pipeline crop and standalone Option C: {non_zero} (out of {diff.size}, {non_zero/diff.size*100:.2f}%)")
            assert non_zero < 10000, f"Difference too large! ({non_zero} pixels)"
            print("Integrated binarise matches Option C parameters perfectly!")
        else:
            print("Standalone Option C crop not found to compare.")
    else:
        print("Output file was not created!")
except Exception as e:
    import traceback
    print(f"Error during binarise run: {e}")
    traceback.print_exc()
    sys.exit(1)
