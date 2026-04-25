from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = _PROJECT_ROOT / "data" / "raw"
PREPROCESSED_DIR = _PROJECT_ROOT / "data" / "preprocessed"
ENHANCED_DIR = _PROJECT_ROOT / "data" / "enhanced"
BINARISED_DIR = _PROJECT_ROOT / "data" / "binarised"
THUMB_DIR = _PROJECT_ROOT / "data" / "thumbnails"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".avif", ".webp"}
THUMB_MAX_PX = 400


@lru_cache(maxsize=1)
def _list_raw_paths() -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for path in RAW_DIR.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    )


def list_raw_images() -> list[Path]:
    """Return cached list of raw images for API reads."""
    return list(_list_raw_paths())


def image_id_for_path(path: Path) -> str:
    """Return a stable UI/API id that stays unique across language folders."""
    return path.relative_to(RAW_DIR).as_posix().replace("/", "__")


def _safe_output_stem(image_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in image_id)


@lru_cache(maxsize=1)
def _raw_path_index() -> dict[str, Path]:
    # Exact folder-aware IDs are unique. Stem-only keys are kept for legacy callers/tests.
    index: dict[str, Path] = {}
    for path in _list_raw_paths():
        index[image_id_for_path(path).lower()] = path
        index.setdefault(path.stem.lower(), path)
    return index


def make_thumbnail(image_id: str) -> Path | None:
    """Return path to cached thumbnail, generating it if needed."""
    import cv2

    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    thumb_path = THUMB_DIR / f"{_safe_output_stem(image_id)}_thumb.jpg"
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
    return _raw_path_index().get(image_id.lower())


def run_stage(image_id: str, stage: str) -> dict:
    if stage == "preprocess":
        return _run_preprocess(image_id)
    if stage == "enhance":
        return _run_enhance(image_id)
    if stage == "binarise":
        return _run_binarise(image_id)
    return {"status": "skipped", "reason": f"Stage '{stage}' not yet implemented"}


def _run_preprocess(image_id: str) -> dict:
    from src.preprocess import preprocess

    raw_path = _find_raw_path(image_id)
    if raw_path is None:
        return {"status": "failed", "error": f"Raw image not found for id '{image_id}'"}

    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PREPROCESSED_DIR / f"{_safe_output_stem(image_id)}_preprocessed.jpg"

    try:
        preprocess(str(raw_path), str(output_path))
        return {
            "status": "done",
            "url": f"/data/preprocessed/{output_path.name}",
        }
    except Exception as exc:
        return {"status": "failed", "error": str(exc)}


def _run_enhance(image_id: str) -> dict:
    from src.enhance import enhance

    # Prefer preprocessed output as input; fall back to raw image
    preprocessed = PREPROCESSED_DIR / f"{_safe_output_stem(image_id)}_preprocessed.jpg"
    src_path = preprocessed if preprocessed.exists() else _find_raw_path(image_id)

    if src_path is None:
        return {"status": "failed", "error": f"No image found for id '{image_id}'"}

    ENHANCED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ENHANCED_DIR / f"{_safe_output_stem(image_id)}_enhanced.jpg"

    try:
        enhance(str(src_path), str(output_path))
        return {
            "status": "done",
            "url": f"/data/enhanced/{output_path.name}",
        }
    except Exception as exc:
        return {"status": "failed", "error": str(exc)}


def _run_binarise(image_id: str) -> dict:
    from src.binarise import binarise

    # Prefer enhanced output as input; fall back to raw image
    enhanced = ENHANCED_DIR / f"{_safe_output_stem(image_id)}_enhanced.jpg"
    src_path = enhanced if enhanced.exists() else _find_raw_path(image_id)

    if src_path is None:
        return {"status": "failed", "error": f"No image found for id '{image_id}'"}

    BINARISED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = BINARISED_DIR / f"{_safe_output_stem(image_id)}_binarised.png"

    try:
        binarise(str(src_path), str(output_path))
        return {
            "status": "done",
            "url": f"/data/binarised/{output_path.name}",
        }
    except Exception as exc:
        return {"status": "failed", "error": str(exc)}
