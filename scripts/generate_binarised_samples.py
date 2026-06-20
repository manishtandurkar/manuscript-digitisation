import shutil
from pathlib import Path
import sys

# Ensure src can be imported
sys.path.append(str(Path(__file__).parent.parent))

from src.binarise import binarise

RAW_DIR = Path("data/raw")
TARGET_DIR = Path("data/binarised_representative_samples")

# The 11 representative images representing different languages and challenges
FOCUS_IMAGES = [
    # Kannada Focus
    "kannada_stone/image2.jpeg",  # Low contrast, weathered
    "kannada_stone/image3.jpeg",  # High contrast, deeply carved

    # Malayalam Focus
    "malayalam_stone/image1.jpeg",  # Low-res, blurred
    "malayalam_stone/image15.jpeg", # Faint, weathered surface

    # Telugu Focus
    "telugu_stone/image2.jpg",      # High contrast, deeply carved text
    "telugu_stone/image40.jpeg",  # Low contrast weathering

    # Tulu Focus
    "tulu_stone/image4.jpeg",        # Faint carvings
    "tulu_stone/image9.JPG",          # Natural texture/colour variations

    # Tamil Focus
    "tamil_stone/tamil_069.jpg",  # Large, low contrast slab
    "tamil_stone/tamil_010.jpg",  # Medium slab, deep shadows
    "tamil_stone/tamil_026.jpg",  # Faint, highly textured surface
]

def main():
    print(f"Creating and organizing binarised representative samples directory: {TARGET_DIR}...")

    # Recreate target directory
    if TARGET_DIR.exists():
        print(f"Cleaning existing files in {TARGET_DIR}...")
        shutil.rmtree(TARGET_DIR)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for rel_path in FOCUS_IMAGES:
        src_path = RAW_DIR / rel_path
        if not src_path.exists():
            print(f"Warning: Source image not found at {src_path}")
            continue

        # Determine output paths inside TARGET_DIR
        rel_p = Path(rel_path)
        lang_dir = TARGET_DIR / rel_p.parent
        lang_dir.mkdir(parents=True, exist_ok=True)

        dest_orig_path = lang_dir / f"{rel_p.stem}_original{rel_p.suffix}"
        dest_bin_path = lang_dir / f"{rel_p.stem}_binarised.png"

        # 1. Copy original image
        try:
            shutil.copy2(src_path, dest_orig_path)
            print(f"Copied original: {rel_path} -> {dest_orig_path.relative_to(TARGET_DIR.parent)}")
        except Exception as exc:
            print(f"Error copying original {rel_path}: {exc}")
            continue

        # 2. Run binarisation on the copied original and save binarised version
        try:
            binarise(str(dest_orig_path), str(dest_bin_path), method="sauvola")
            print(f"Binarised image saved: {dest_bin_path.relative_to(TARGET_DIR.parent)}")
            success_count += 1
        except Exception as exc:
            print(f"Error binarising {dest_orig_path}: {exc}")

    print(f"\nCompleted! Successfully processed {success_count} representative pairs under {TARGET_DIR}.")

if __name__ == "__main__":
    main()
