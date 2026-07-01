import shutil
from pathlib import Path

src = Path("data/binarised_representative_samples/tamil_stone/tamil_026_binarised_FIXED.png")
dest = Path(r"C:\Users\nanda_4h6zihz\.gemini\antigravity-ide\brain\9bb206b2-f39c-4a1f-befa-83279631baab\tamil_026_binarised_FIXED.png")

if src.exists():
    shutil.copy(str(src), str(dest))
    print(f"Copied {src} to {dest}")
else:
    print(f"Source file {src} does not exist")
