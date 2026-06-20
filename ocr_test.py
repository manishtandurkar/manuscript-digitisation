import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.ocr import transcribe

img_path = "data/binarised_representative_samples/kannada_stone/image2_final_test3.png"
out_path = "data/binarised_representative_samples/kannada_stone/image2_transcription.json"

result = transcribe(img_path, script="kannada", output_path=out_path)

print(f"Script: {result['script']}")
print(f"Engine used: {result['engine_used']}")
print(f"Confidence: {result['overall_confidence']} ({result['confidence_status']})")
print(f"Number of lines detected: {len(result['lines'])}")
print(f"\nFull text:\n{result['text']}")