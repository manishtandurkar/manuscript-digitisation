import glob
import os
from pathlib import Path
import cv2
import numpy as np

brain_dir = r"C:\Users\nanda_4h6zihz\.gemini\antigravity-ide\brain\9bb206b2-f39c-4a1f-befa-83279631baab"
files = glob.glob(os.path.join(brain_dir, "media__*"))
files.sort(key=os.path.getmtime)

print(f"Found {len(files)} media files in brain directory:")
for f in files:
    mtime = os.path.getmtime(f)
    from datetime import datetime
    dt = datetime.fromtimestamp(mtime).isoformat()
    img = cv2.imread(f)
    if img is not None:
        print(f"File: {os.path.basename(f)} | Time: {dt} | Shape: {img.shape}")
        # print some statistics about colors
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        unique, counts = np.unique(gray, return_counts=True)
        # print top 3 gray values
        sorted_indices = np.argsort(-counts)
        top_vals = [f"{unique[idx]}:{counts[idx]} ({counts[idx]/gray.size*100:.1f}%)" for idx in sorted_indices[:3]]
        print(f"  Top values: {', '.join(top_vals)}")
