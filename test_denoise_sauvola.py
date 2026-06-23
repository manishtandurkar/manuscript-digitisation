import cv2
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from src.binarise import remove_noise_blobs

def main():
    sauvola_bin = cv2.imread("tune_img334_out/sauvola_bin.png", cv2.IMREAD_GRAYSCALE)
    if sauvola_bin is None:
        print("Failed to load sauvola_bin.png")
        return

    # Let's try different min_size and min_length values
    for min_size in [20, 50, 80, 120, 200]:
        for min_length in [10, 15, 20, 25, 30]:
            cleaned = remove_noise_blobs(sauvola_bin, min_size=min_size, min_length=min_length)
            mean_val = cleaned.mean()
            # Save if it's not empty and not full
            if 1.0 < mean_val < 250.0:
                out_path = f"tune_img334_out/clean_sz{min_size}_len{min_length}.png"
                cv2.imwrite(out_path, cleaned)
                print(f"Saved {out_path} with mean {mean_val:.2f}")

if __name__ == "__main__":
    main()
