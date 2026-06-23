import os
import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import cv2
from src.binarise import binarise

def main():
    raw_dir = Path("data/raw/tamil_stone")
    output_dir = Path("data/binarised_tamizh")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Supported suffixes
    suffixes = {".jpg", ".jpeg", ".png"}
    img_paths = sorted([p for p in raw_dir.glob("*") if p.suffix.lower() in suffixes])
    
    total = len(img_paths)
    print(f"Found {total} Tamil raw images to process.")
    
    success_count = 0
    fail_count = 0
    
    start_time = time.time()
    for idx, img_path in enumerate(img_paths, 1):
        out_path = output_dir / f"{img_path.stem}_binarised.png"
        print(f"[{idx}/{total}] Processing {img_path.name} -> {out_path.name}...")
        
        step_start = time.time()
        try:
            binarise(str(img_path), str(out_path), method="sauvola")
            success_count += 1
            duration = time.time() - step_start
            print(f"   Success! ({duration:.2f}s)")
        except Exception as e:
            fail_count += 1
            print(f"   Failed: {e}")
            
    total_time = time.time() - start_time
    print("\n==============================================")
    print(f"Finished processing all Tamil images.")
    print(f"Successfully processed: {success_count}/{total}")
    print(f"Failed: {fail_count}/{total}")
    print(f"Total time elapsed: {total_time:.2f}s (Avg {total_time/total:.2f}s per image)")
    print("==============================================")

if __name__ == "__main__":
    main()
