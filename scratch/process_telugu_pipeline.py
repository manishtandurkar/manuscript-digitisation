import os
import sys
import shutil
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from api.pipeline import run_stage, list_raw_images, image_id_for_path, _safe_output_stem

def main():
    # 1. Filter Telugu raw images
    all_raw = list_raw_images()
    telugu_raw = [p for p in all_raw if "telugu_stone" in p.as_posix().lower()]
    
    total = len(telugu_raw)
    print(f"Found {total} Telugu raw images in the backend database.")
    
    # 2. Create target directory
    target_dir = Path("data/binarised_telugu")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Process each image end-to-end
    start_time = time.time()
    success_count = 0
    
    for idx, path in enumerate(telugu_raw, 1):
        image_id = image_id_for_path(path)
        stem = _safe_output_stem(image_id)
        print(f"[{idx}/{total}] Processing image ID: {image_id}...")
        
        try:
            # Step A: Preprocess
            print("   Running Preprocess...")
            prep_res = run_stage(image_id, "preprocess")
            if prep_res.get("status") == "failed":
                raise ValueError(f"Preprocess failed: {prep_res.get('error')}")
                
            # Step B: Enhance (Mild Mode)
            print("   Running Enhance (Mild)...")
            enh_res = run_stage(image_id, "enhance", {"mode": "mild"})
            if enh_res.get("status") == "failed":
                raise ValueError(f"Enhance failed: {enh_res.get('error')}")
                
            # Step C: Binarise (Sauvola Method)
            print("   Running Binarise (Sauvola)...")
            bin_res = run_stage(image_id, "binarise", {"method": "sauvola"})
            if bin_res.get("status") == "failed":
                raise ValueError(f"Binarise failed: {bin_res.get('error')}")
                
            # Copy binarised output to data/binarised_telugu/
            binarised_src = Path("data/binarised") / f"{stem}_binarised.png"
            if binarised_src.exists():
                # Destination name: clean raw image stem (e.g. image1_binarised.png)
                dest_path = target_dir / f"{path.stem}_binarised.png"
                shutil.copy(str(binarised_src), str(dest_path))
                print(f"   Saved to {dest_path}")
                success_count += 1
            else:
                print(f"   Warning: binarised output not found at {binarised_src}")
                
        except Exception as e:
            print(f"   Failed: {e}")
            
    total_time = time.time() - start_time
    print("\n==============================================")
    print(f"Completed processing Telugu pipeline.")
    print(f"Successfully processed & copied: {success_count}/{total} images.")
    print(f"Binarised outputs saved in: {target_dir.resolve()}")
    print(f"Total time elapsed: {total_time:.2f}s (Avg {total_time/total:.2f}s per image)")
    print("==============================================")

if __name__ == "__main__":
    main()
