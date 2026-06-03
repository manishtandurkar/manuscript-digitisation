import os
from pathlib import Path
import cv2
import numpy as np

RAW_DIR = Path("data/raw")

def analyze_image(path: Path):
    try:
        img = cv2.imread(str(path))
        if img is None:
            return None
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        mean_val = float(gray.mean())
        std_val = float(gray.std())
        sat_mean = float(hsv[:, :, 1].mean())
        hue_mean = float(hsv[:, :, 0].mean())
        
        return {
            "path": path,
            "filename": path.name,
            "folder": path.parent.name,
            "width": w,
            "height": h,
            "mean": round(mean_val, 1),
            "std": round(std_val, 1),
            "saturation": round(sat_mean, 1),
            "hue": round(hue_mean, 1),
            "size_kb": round(os.path.getsize(path) / 1024, 1)
        }
    except Exception as exc:
        return None

def main():
    results = []
    for root, _, files in os.walk(RAW_DIR):
        for f in files:
            path = Path(root) / f
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".avif", ".webp"}:
                meta = analyze_image(path)
                if meta:
                    results.append(meta)
                    
    print(f"Total analyzed images: {len(results)}\n")
    
    # Group by folder
    by_folder = {}
    for r in results:
        by_folder.setdefault(r["folder"], []).append(r)
        
    for folder, items in by_folder.items():
        print(f"--- Folder: {folder} ({len(items)} images) ---")
        # Sort items to find extremes
        items_sorted_by_size = sorted(items, key=lambda x: x["width"] * x["height"])
        items_sorted_by_std = sorted(items, key=lambda x: x["std"])
        items_sorted_by_sat = sorted(items, key=lambda x: x["saturation"])
        
        print("Representative choices (Smallest, Largest, Low-contrast, High-contrast, High-saturation):")
        choices = {
            "Smallest Resolution": items_sorted_by_size[0],
            "Largest Resolution": items_sorted_by_size[-1],
            "Lowest Contrast (Standard Dev)": items_sorted_by_std[0],
            "Highest Contrast (Standard Dev)": items_sorted_by_std[-1],
            "Highest Saturation (Warm Tones)": items_sorted_by_sat[-1],
        }
        
        seen_filenames = set()
        for label, choice in choices.items():
            fn = choice["filename"]
            if fn not in seen_filenames:
                seen_filenames.add(fn)
                print(f"  * {label}: {fn} ({choice['width']}x{choice['height']}, Mean={choice['mean']}, Std={choice['std']}, Sat={choice['saturation']}, Size={choice['size_kb']} KB)")

if __name__ == "__main__":
    main()
