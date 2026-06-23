import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from src.binarise import binarise

input_path = "data/binarised_representative_samples/tamil_stone/tamil_026_original.jpg"
output_path = "scratch/test_pipeline_output.png"

try:
    print(f"Calling binarise on {input_path}...")
    result = binarise(input_path, output_path)
    print("Success calling binarise!")
    
    out_p = Path(output_path)
    if out_p.exists():
        img = cv2.imread(str(out_p))
        print(f"Output exists! Shape: {img.shape}")
        # Count black and white pixels (main pipeline always returns white text on black background)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        total = gray.size
        black = np.sum(gray == 0)
        white = np.sum(gray == 255)
        print(f"Pipeline output stats: black={black} ({black/total*100:.2f}%), white={white} ({white/total*100:.2f}%)")
        
        # Verify it matches the target (inverted)
        # Let's count how many pixels are different from data/binarised/tamil_stone__tamil_026_jpg_binarised.png
        ref_path = "data/binarised/tamil_stone__tamil_026_jpg_binarised.png"
        ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        if ref is not None:
            diff = cv2.absdiff(gray, ref)
            non_zero = np.count_nonzero(diff)
            print(f"Difference against reference binarised image: {non_zero} pixels ({non_zero/total*100:.2f}%)")
            assert non_zero < 2000, "Difference too large!"
            print("Integrated routing matches the high-quality target binarisation!")
        else:
            print("Reference binarised image not found to compare.")
    else:
        print("Output file was not created!")
except Exception as e:
    import traceback
    print(f"Error during binarise run: {e}")
    traceback.print_exc()
