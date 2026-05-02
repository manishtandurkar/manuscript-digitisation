import cv2
import numpy as np
from pathlib import Path

# Change all four paths
raw_path       = "C:\\6th semester EL's\\Interdisciplinary project\\Implementation\\manuscript-digitisation\\data\\raw\\malayalam_stone\\image1.jpeg"
preprocessed_path = "C:\\6th semester EL's\\Interdisciplinary project\\Implementation\\manuscript-digitisation\\data\\preprocessed\\malayalam_stone__image1_jpeg_preprocessed.jpg"
enhanced_path  = "C:\\6th semester EL's\\Interdisciplinary project\\Implementation\\manuscript-digitisation\\data\\enhanced\\malayalam_stone__image1_jpeg_enhanced_dstretch.jpg"

out = Path("stage_test")
out.mkdir(exist_ok=True)

def test_stage(label, path):
    img = cv2.imread(path)
    if img is None:
        print(f"{label}: COULD NOT READ IMAGE at {path}")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(out/f"{label}_0_input.png"), img)
    cv2.imwrite(str(out/f"{label}_1_gray.png"), gray)
    
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 5
    )
    binary = cv2.bitwise_not(binary)
    cv2.imwrite(str(out/f"{label}_2_binary.png"), binary)
    
    print(f"\n{label}:")
    print(f"  image size : {img.shape}")
    print(f"  gray  mean : {gray.mean():.1f}  min:{gray.min()}  max:{gray.max()}")
    print(f"  fg%%       : {cv2.countNonZero(binary)/binary.size*100:.1f}%%")

test_stage("1_raw",          raw_path)
test_stage("2_preprocessed", preprocessed_path)
test_stage("3_enhanced",     enhanced_path)

print("\nDone. Check stage_test folder.")