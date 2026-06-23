import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import easyocr
import sys

def get_font_for_script(script: str, font_size: int):
    """
    Returns a loaded TrueType font for the given script from Windows system fonts,
    falling back to Nirmala (common Unicode font) or default if not found.
    """
    font_paths = {
        "kannada": [r"C:\Windows\Fonts\tunga.ttf", r"C:\Windows\Fonts\Nirmala.ttf"],
        "tamil": [r"C:\Windows\Fonts\latha.ttf", r"C:\Windows\Fonts\Nirmala.ttf"],
        "telugu": [r"C:\Windows\Fonts\gautami.ttf", r"C:\Windows\Fonts\Nirmala.ttf"],
        "malayalam": [r"C:\Windows\Fonts\kartika.ttf", r"C:\Windows\Fonts\Nirmala.ttf"],
        "tulu": [r"C:\Windows\Fonts\tunga.ttf", r"C:\Windows\Fonts\Nirmala.ttf"],
    }
    paths = font_paths.get(script.lower(), [r"C:\Windows\Fonts\Nirmala.ttf"])
    for p in paths:
        if Path(p).exists():
            return ImageFont.truetype(p, font_size)
    return ImageFont.load_default()

def find_best_font_size(draw, text, script, target_w, target_h):
    """
    Finds the largest font size (up to a limit) where the rendered text
    fits within the target bounding box width and height.
    """
    best_size = 10
    # Search font size space iteratively
    for size in range(12, 150, 2):
        try:
            font = get_font_for_script(script, size)
            bbox = draw.textbbox((0, 0), text, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w <= target_w and h <= target_h:
                best_size = size
            else:
                break
        except Exception:
            break
    return best_size

def reconstruct_ocr(image_path: str):
    path = Path(image_path)
    if not path.exists():
        print(f"Error: File not found at {image_path}")
        return
        
    print(f"\n--- Running OCR-Guided Reconstruction for {path.name} ---")
    
    # 1. Detect Script/Language from folder path
    parent_dir_name = path.parent.name.lower()
    
    # Defaults
    ocr_langs = ["kn"]
    script_name = "kannada"
    
    if "tamil" in parent_dir_name:
        ocr_langs = ["ta"]
        script_name = "tamil"
    elif "telugu" in parent_dir_name:
        ocr_langs = ["te"]
        script_name = "telugu"
    elif "malayalam" in parent_dir_name:
        ocr_langs = ["ml"]
        script_name = "malayalam"
    elif "tulu" in parent_dir_name:
        ocr_langs = ["kn"]
        script_name = "tulu"
        
    print(f"Detected Script: {script_name.upper()} (EasyOCR lang code: {ocr_langs})")
    
    # 2. Load the original color/gray image
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("Failed to load image.")
        return
        
    H, W = img.shape[:2]
    
    # 3. Initialise EasyOCR Reader (CPU mode to avoid CUDA conflicts)
    print("Initialising EasyOCR Reader (running on CPU)...")
    reader = easyocr.Reader(ocr_langs, gpu=False, verbose=False)
    
    # 4. Run text detection and recognition
    print("Running OCR on image. This may take a few seconds...")
    results = reader.readtext(img, detail=1)
    print(f"OCR finished. Found {len(results)} word blocks.")
    
    # 5. Create blank black canvas for rendering
    canvas = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(canvas)
    
    valid_render_count = 0
    
    # 6. Render recognized text onto the canvas inside the bounding boxes
    for bbox, text, conf in results:
        text = (text or "").strip()
        if not text or conf < 0.20:  # Exclude very low confidence noise detections
            continue
            
        # bbox is: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        x, y = int(min(xs)), int(min(ys))
        w = int(max(xs) - min(xs))
        h = int(max(ys) - min(ys))
        
        # Skip weirdly small detections
        if w < 6 or h < 6:
            continue
            
        # Find best font size to fit the detected bounding box
        font_size = find_best_font_size(draw, text, script_name, w, h)
        font = get_font_for_script(script_name, font_size)
        
        # Render the text in white
        draw.text((x, y), text, font=font, fill=255)
        valid_render_count += 1
        
    # Convert PIL image back to numpy array for saving
    output_np = np.array(canvas)
    
    # Save output in same directory
    output_path = path.parent / f"{path.stem}_binarised_ocr_reconstructed.png"
    cv2.imwrite(str(output_path), output_np)
    
    print(f"Successfully rendered and stamped {valid_render_count} text blocks.")
    print(f"Saved reconstructed image to: {output_path}")

if __name__ == "__main__":
    # Default path if none provided as argument
    default_img = r"C:\6th semester EL's\Interdisciplinary project\Implementation\manuscript-digitisation\data\binarised_representative_samples\kannada_stone\image2_original.jpeg"
    
    target_path = sys.argv[1] if len(sys.argv) > 1 else default_img
    reconstruct_ocr(target_path)
