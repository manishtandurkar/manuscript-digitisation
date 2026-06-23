import cv2
import numpy as np
from pathlib import Path
from src.binarise import binarise_stone, detect_document_type

img_path = r"data/raw/tamil_stone/IMG_3924.jpg"
p = Path(img_path)

if not p.exists():
    # Let's search the workspace for any file containing IMG_3924
    print(f"File {img_path} does not exist. Searching...")
    import os
    for root, dirs, files in os.walk("."):
        for f in files:
            if "IMG_3924" in f:
                p = Path(os.path.join(root, f))
                print(f"Found file: {p}")
                break

if p.exists():
    img = cv2.imread(str(p))
    print(f"Image shape: {img.shape}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    print(f"Mean brightness: {gray.mean():.2f}")
    print(f"Std dev: {gray.std():.2f}")
    print(f"Min/Max: {gray.min()}/{gray.max()}")
    
    # Run detect_document_type
    doc_type = detect_document_type(img, img_path=p)
    print(f"Detected doc type: {doc_type}")
    
    # Run binarise_stone
    binary = binarise_stone(img)
    white_pct = (binary == 255).mean() * 100
    print(f"Binarise stone output: white (text) pixels = {white_pct:.2f}%")
else:
    print("Could not find IMG_3924 image anywhere in the workspace.")
