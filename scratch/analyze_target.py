import cv2
import numpy as np
from pathlib import Path

original_path = r"data/binarised_representative_samples/tamil_stone/tamil_026_original.jpg"
fixed_path = r"data/binarised_representative_samples/tamil_stone/tamil_026_binarised_FIXED.png"
target_path = r"C:\Users\nanda_4h6zihz\.gemini\antigravity-ide\brain\9bb206b2-f39c-4a1f-befa-83279631baab\media__1782218721649.png"

def check_img(name, path):
    p = Path(path)
    if not p.exists():
        print(f"{name} does not exist at {path}")
        return None
    img = cv2.imread(str(p))
    if img is None:
        print(f"Failed to load {name} from {path}")
        return None
    print(f"{name}: shape={img.shape}, dtype={img.dtype}, unique_pixels={np.unique(img)}")
    return img

orig = check_img("Original", original_path)
fixed = check_img("Fixed", fixed_path)
target = check_img("Target", target_path)

if fixed is not None:
    # Let's count black and white pixels in fixed
    black_count = np.sum(fixed == 0)
    white_count = np.sum(fixed == 255)
    total = fixed.size
    print(f"Fixed: black={black_count} ({black_count/total*100:.2f}%), white={white_count} ({white_count/total*100:.2f}%)")

if target is not None:
    black_count = np.sum(target == 0)
    white_count = np.sum(target == 255)
    total = target.size
    print(f"Target: black={black_count} ({black_count/total*100:.2f}%), white={white_count} ({white_count/total*100:.2f}%)")
