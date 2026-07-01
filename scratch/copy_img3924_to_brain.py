import shutil
from pathlib import Path

brain_dir = Path(r"C:\Users\nanda_4h6zihz\.gemini\antigravity-ide\brain\9bb206b2-f39c-4a1f-befa-83279631baab")

# Files to copy
files_to_copy = [
    ("scratch/img3924_crop_gray.jpg", "img3924_crop_gray.jpg"),
    ("scratch/img3924_crop_cleaned_default.png", "img3924_crop_cleaned_default.png"),
    ("scratch/img3924_crop_m21_ws101_k35_inv.png", "img3924_crop_m21_ws101_k35_inv.png"),
    ("scratch/img3924_crop_m21_ws51_k25_inv.png", "img3924_crop_m21_ws51_k25_inv.png"),
    ("scratch/img3924_crop_m31_ws101_k25_inv.png", "img3924_crop_m31_ws101_k25_inv.png")
]

for src_name, dest_name in files_to_copy:
    src_path = Path(src_name)
    dest_path = brain_dir / dest_name
    if src_path.exists():
        shutil.copy(str(src_path), str(dest_path))
        print(f"Copied {src_name} to {dest_path}")
    else:
        print(f"File {src_name} does not exist")
