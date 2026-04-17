from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def ensure_parent_dir(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def save_image(path: str | Path, image_bgr: np.ndarray, jpeg_quality: int = 95) -> Path:
    """Save a BGR OpenCV image as JPEG via PIL."""
    output_path = ensure_parent_dir(path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_rgb)
    image.save(str(output_path), quality=jpeg_quality, subsampling=0)
    return output_path
