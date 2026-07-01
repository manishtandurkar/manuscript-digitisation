import cv2
import numpy as np
from pathlib import Path

img1_path = r"C:\Users\nanda_4h6zihz\.gemini\antigravity-ide\brain\9bb206b2-f39c-4a1f-befa-83279631baab\media__1782218721649.png"
img2_path = r"C:\Users\nanda_4h6zihz\.gemini\antigravity-ide\brain\9bb206b2-f39c-4a1f-befa-83279631baab\media__1782219053166.png"

def check_img(name, path):
    p = Path(path)
    if not p.exists():
        print(f"{name} does not exist")
        return
    img = cv2.imread(str(p))
    if img is None:
        print(f"Failed to load {name}")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    total = gray.size
    black = np.sum(gray == 0)
    white = np.sum(gray == 255)
    other = total - black - white
    print(f"{name}: shape={img.shape}, black={black} ({black/total*100:.2f}%), white={white} ({white/total*100:.2f}%), other={other} ({other/total*100:.2f}%)")

check_img("media__1782218721649", img1_path)
check_img("media__1782219053166", img2_path)
