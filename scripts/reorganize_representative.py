import shutil
from pathlib import Path

# Paths
RAW_DIR = Path("data/raw")
REP_DIR = Path("data/representative_raw")

# Mapping of selected focus images: (source path relative to data/raw, destination path relative to data/representative_raw)
FOCUS_IMAGES = [
    # Kannada Focus
    ("kannada_stone/image2.jpeg", "kannada_stone/image2.jpeg"),  # Low contrast, weathered
    ("kannada_stone/image3.jpeg", "kannada_stone/image3.jpeg"),  # High contrast, deeply carved

    # Malayalam Focus
    ("malayalam_stone/image1.jpeg", "malayalam_stone/image1.jpeg"),  # Low-res, blurred
    ("malayalam_stone/image15.jpeg", "malayalam_stone/image15.jpeg"), # Faint, weathered surface

    # Telugu Focus
    ("telugu_stone/image2.jpg", "telugu_stone/image2.jpg"),      # High contrast, deeply carved text
    ("telugu_stone/image40.jpeg", "telugu_stone/image40.jpeg"),  # Low contrast weathering

    # Tulu Focus
    ("tulu_stone/image4.jpeg", "tulu_stone/image4.jpeg"),        # Faint carvings
    ("tulu_stone/image9.JPG", "tulu_stone/image9.JPG"),          # Natural texture/colour variations

    # Tamil Focus
    ("tamil_stone/tamil_069.jpg", "tamil_stone/tamil_069.jpg"),  # Large, low contrast slab (21MB)
    ("tamil_stone/tamil_010.jpg", "tamil_stone/tamil_010.jpg"),  # Medium slab, deep shadows
    ("tamil_stone/tamil_026.jpg", "tamil_stone/tamil_026.jpg"),  # Faint, highly textured surface
]

def main():
    print("Reorganizing representative_raw directory...")

    # 1. Remove pre-existing stuff in data/representative_raw
    if REP_DIR.exists():
        print(f"Cleaning up existing contents in {REP_DIR}...")
        for child in REP_DIR.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        REP_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Copy the selected focus images
    copied_count = 0
    for src_rel, dest_rel in FOCUS_IMAGES:
        src_path = RAW_DIR / src_rel
        dest_path = REP_DIR / dest_rel

        if not src_path.exists():
            print(f"Warning: Source image not found at {src_path}")
            continue

        # Create parent directory if needed
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy file
        shutil.copy2(src_path, dest_path)
        print(f"Copied: {src_rel} -> {dest_rel}")
        copied_count += 1

    print(f"\nSuccessfully reorganized. Copied {copied_count} representative focus images.")

if __name__ == "__main__":
    main()
