from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = _PROJECT_ROOT / "data" / "raw"
ENHANCED_DIR = _PROJECT_ROOT / "data" / "enhanced"
THUMB_DIR = _PROJECT_ROOT / "data" / "thumbnails"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
THUMB_MAX_PX = 400


def make_thumbnail(image_id: str) -> Path | None:
    """Return path to cached thumbnail, generating it if needed."""
    import cv2

    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    thumb_path = THUMB_DIR / f"{image_id}_thumb.jpg"
    if thumb_path.exists():
        return thumb_path

    raw_path = _find_raw_path(image_id)
    if raw_path is None:
        return None

    img = cv2.imread(str(raw_path))
    if img is None:
        return None

    h, w = img.shape[:2]
    scale = THUMB_MAX_PX / max(h, w)
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(thumb_path), small, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return thumb_path


def _find_raw_path(image_id: str) -> Path | None:
    for path in RAW_DIR.rglob("*"):
        if path.is_file() and path.stem.lower() == image_id.lower() and path.suffix.lower() in IMAGE_SUFFIXES:
            return path
    return None


def run_stage(image_id: str, stage: str) -> dict:
    if stage == "preprocess":
        return _run_preprocess(image_id)
    if stage == "enhance":
        return _run_enhance(image_id)
    return {"status": "skipped", "reason": f"Stage '{stage}' not yet implemented"}


def _run_preprocess(image_id: str) -> dict:
    from src.preprocess import preprocess, build_output_path

    raw_path = _find_raw_path(image_id)
    if raw_path is None:
        return {"status": "failed", "error": f"Raw image not found for id '{image_id}'"}

    ENHANCED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = build_output_path(raw_path, ENHANCED_DIR)

    try:
        preprocess(str(raw_path), str(output_path))
        return {
            "status": "done",
            "url": f"/data/enhanced/{output_path.name}",
        }
    except Exception as exc:
        return {"status": "failed", "error": str(exc)}


def _run_enhance(image_id: str) -> dict:
    from src.enhance import enhance

    # Prefer preprocessed output as input; fall back to raw image
    preprocessed = ENHANCED_DIR / f"{image_id}_preprocessed.jpg"
    src_path = preprocessed if preprocessed.exists() else _find_raw_path(image_id)

    if src_path is None:
        return {"status": "failed", "error": f"No image found for id '{image_id}'"}

    ENHANCED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ENHANCED_DIR / f"{image_id}_enhanced.jpg"

    try:
        enhance(str(src_path), str(output_path))
        return {
            "status": "done",
            "url": f"/data/enhanced/{output_path.name}",
        }
    except Exception as exc:
        return {"status": "failed", "error": str(exc)}
