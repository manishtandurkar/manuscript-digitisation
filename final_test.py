import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.binarise import binarise

src_path = "data/binarised_representative_samples/kannada_stone/image2_original.jpeg"
out_path = "data/binarised_representative_samples/kannada_stone/image2_final_test3.png"

binarise(src_path, out_path, method="sauvola")
print(f"Done: {out_path}")