import cv2
from pathlib import Path

raw_path = Path("data/raw/tamil_stone/tamil_026.jpg")
if raw_path.exists():
    img = cv2.imread(str(raw_path))
    print(f"Raw image shape: {img.shape}")
else:
    print("Raw image does not exist at", raw_path)
