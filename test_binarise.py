import sys
import argparse
from pathlib import Path
import cv2
import numpy as np

# Ensure src can be imported
sys.path.append(str(Path(__file__).parent))

from src.binarise import (
    binarise_stone,
    binarise_palm_leaf,
    detect_document_type,
    detect_rubbing,
    binarise_rubbing,
    remove_noise_blobs
)

def main():
    parser = argparse.ArgumentParser(description="Standalone Binarisation & Segmentation Feasibility Tester")
    parser.add_argument("input_image", help="Path to input image")
    parser.add_argument("--force-type", choices=["stone", "palm_leaf"], default=None, help="Force document type classification")
    parser.add_argument("--out-dir", default="./test_output", help="Directory to save test outputs")
    args = parser.parse_args()

    img_path = Path(args.input_image)
    if not img_path.exists():
        print(f"Error: Input image not found at {img_path}")
        sys.exit(1)

    # Read image safely on Windows
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: Could not read image {img_path}")
        sys.exit(1)

    # 1. Document Type Detection
    detected_type = detect_document_type(img)
    is_rub = detect_rubbing(img)
    print(f"Detected document type: {detected_type} (is_rubbing: {is_rub})")

    # Determine paths based on force-type or detection
    if args.force_type:
        doc_type = args.force_type
        print(f"Forcing document type to: {doc_type}")
    else:
        doc_type = detected_type
        print(f"Using detected type: {doc_type}")

    # 2. Binarise
    if doc_type == "palm_leaf":
        binary = binarise_palm_leaf(img)
    else:
        # Check rubbing if not forced
        if not args.force_type and is_rub:
            print("Applying binarise_rubbing path...")
            binary = binarise_rubbing(img)
        else:
            print("Applying binarise_stone path...")
            binary = binarise_stone(img)

    # Polar check and noise blob cleanup (mirroring public dispatcher)
    if binary.mean() >= 127:
        binary = cv2.bitwise_not(binary)

    if doc_type == "palm_leaf":
        binary = remove_noise_blobs(binary, min_size=8, min_length=15)
    else:
        binary = remove_noise_blobs(binary, min_size=80, min_length=25)

    # 3. Save Binarised Output
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_out_path = out_dir / f"{img_path.stem}_binarised.png"
    cv2.imwrite(str(bin_out_path), binary)
    print(f"Saved binarised image to: {bin_out_path}")

    # 4. Connected Components & Feasibility Metrics
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Analyze components (excluding background label 0)
    components = []
    shorter = min(binary.shape[:2])
    
    # Bins for classification:
    # Tiny: noise < 80 pixels (or < min_size)
    # Small: 80 <= area < 300
    # Medium: 300 <= area < 2000 (typical single glyphs)
    # Huge: area >= 2000 or w > shorter // 8 or h > shorter // 8 (merged blobs)
    tiny_cnt = 0
    small_cnt = 0
    med_cnt = 0
    huge_cnt = 0

    debug_img = img.copy()

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        
        components.append(area)
        
        # Classification
        if area < 80:
            tiny_cnt += 1
            # Draw Yellow box for noise
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 255), 1)
        elif area >= 2000 or w > (shorter // 8) or h > (shorter // 8):
            huge_cnt += 1
            # Draw Red box for merged blobs
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        else:
            # Medium and small glyph candidates
            if area < 300:
                small_cnt += 1
            else:
                med_cnt += 1
            # Draw Green box for good glyph candidates
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 1)

    areas = np.array(components) if components else np.array([0])
    
    # 5. Output Results
    print("\n--- Segmentation Feasibility Summary ---")
    print(f"total_components: {len(components)}")
    print(f"tiny_count (noise): {tiny_cnt}")
    print(f"small_count (fragments): {small_cnt}")
    print(f"medium_count (single glyphs): {med_cnt}")
    print(f"huge_count (merged blobs): {huge_cnt}")
    print(f"area_min: {areas.min()}")
    print(f"area_max: {areas.max()}")
    print(f"area_median: {np.median(areas):.1f}")
    
    # Save debug image
    debug_out_path = out_dir / f"{img_path.stem}_components_debug.png"
    cv2.imwrite(str(debug_out_path), debug_img)
    print(f"Saved debug image to: {debug_out_path}")

if __name__ == "__main__":
    main()
