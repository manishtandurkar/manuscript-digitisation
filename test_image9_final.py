import cv2
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from src.binarise import binarise, detect_document_type

def main():
    img_path = r"C:\6th semester EL's\Interdisciplinary project\Implementation\manuscript-digitisation\data\raw\malayalam_stone\image9.png"
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    
    # 1. Document type detection check
    doc_type = detect_document_type(img, img_path=img_path)
    print(f"Detected document type: {doc_type}")
    assert doc_type == "stone", f"Expected stone, but got {doc_type}"

    # 2. Run the main binarise dispatcher
    out_dir = Path("tune_img334_out")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "image9_binarise_integrated.png"
    
    binary = binarise(img_path, str(out_path), method="sauvola")
    print(f"Success! Output mean: {binary.mean():.2f}")
    
    # Check if corner pixels are black (0) after flood fill
    h, w = binary.shape[:2]
    print(f"Corner pixels: TL={binary[0,0]}, TR={binary[0,w-1]}, BL={binary[h-1,0]}, BR={binary[h-1,w-1]}")

if __name__ == "__main__":
    main()
