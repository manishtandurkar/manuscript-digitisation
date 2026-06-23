from pathlib import Path
import cv2

p = Path("scratch/binary_target_cropped.png")
if p.exists():
    img = cv2.imread(str(p))
    print(f"File exists! Shape: {img.shape}")
else:
    print("File does not exist yet")
