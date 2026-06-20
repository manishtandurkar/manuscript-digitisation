import shutil
from pathlib import Path

# Paths
RAW_DIR = Path("data/raw")
SAMPLES_DIR = Path("data/representative_samples")

# Mapping of selected focus images (source path relative to data/raw, destination path relative to data/representative_samples)
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
    ("tamil_stone/tamil_069.jpg", "tamil_stone/tamil_069.jpg"),  # Large, low contrast slab
    ("tamil_stone/tamil_010.jpg", "tamil_stone/tamil_010.jpg"),  # Medium slab, deep shadows
    ("tamil_stone/tamil_026.jpg", "tamil_stone/tamil_026.jpg"),  # Faint, highly textured surface
]

def main():
    print(f"Creating and organizing {SAMPLES_DIR}...")

    # Recreate the folder structure cleanly
    if SAMPLES_DIR.exists():
        print(f"Cleaning existing files in {SAMPLES_DIR}...")
        shutil.rmtree(SAMPLES_DIR)
    
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    for src_rel, dest_rel in FOCUS_IMAGES:
        src_path = RAW_DIR / src_rel
        dest_path = SAMPLES_DIR / dest_rel

        if not src_path.exists():
            print(f"Warning: Source image not found at {src_path}")
            continue

        # Create parent directory in destination
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy the file
        shutil.copy2(src_path, dest_path)
        print(f"Copied: {src_rel} -> {dest_rel}")
        copied_count += 1

    print(f"\nSuccessfully copied {copied_count} representative focus images to {SAMPLES_DIR}.")

if __name__ == "__main__":
    main()
