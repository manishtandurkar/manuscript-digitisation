from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
from PIL import Image


EXIF_POINTER_TAGS = {34665, 34853, 40965}


def ensure_parent_dir(path: str | Path) -> Path:
    """Create the parent directory for a file path if it does not exist."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _sanitise_exif(image: Image.Image) -> Image.Exif | None:
    exif = image.getexif()
    if not exif:
        return None

    cleaned = Image.Exif()
    for tag, value in exif.items():
        if tag not in EXIF_POINTER_TAGS:
            cleaned[tag] = value
    return cleaned


def read_image_metadata(path: str | Path) -> Dict[str, Any]:
    """Read basic metadata that can be forwarded when saving processed images."""
    with Image.open(path) as image:
        return {
            "exif": _sanitise_exif(image),
            "icc_profile": image.info.get("icc_profile"),
            "dpi": image.info.get("dpi"),
        }


def save_image(
    path: str | Path,
    image_bgr: np.ndarray,
    metadata: Dict[str, Any] | None = None,
    jpeg_quality: int = 95,
) -> Path:
    """Save a BGR OpenCV image via PIL while forwarding available metadata."""
    metadata = metadata or {}
    output_path = ensure_parent_dir(path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_rgb)

    save_kwargs: Dict[str, Any] = {}
    if metadata.get("exif") is not None:
        save_kwargs["exif"] = metadata["exif"]
    if metadata.get("icc_profile") is not None:
        save_kwargs["icc_profile"] = metadata["icc_profile"]
    if metadata.get("dpi") is not None:
        save_kwargs["dpi"] = metadata["dpi"]

    suffix = output_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        save_kwargs["quality"] = jpeg_quality
        save_kwargs["subsampling"] = 0
    elif suffix in {".tif", ".tiff"}:
        save_kwargs["compression"] = "tiff_lzw"

    image.save(output_path, **save_kwargs)
    return output_path
