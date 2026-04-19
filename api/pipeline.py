from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = _PROJECT_ROOT / "data" / "raw"
ENHANCED_DIR = _PROJECT_ROOT / "data" / "enhanced"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _find_raw_path(image_id: str) -> Path | None:
    for path in RAW_DIR.rglob("*"):
        if path.is_file() and path.stem == image_id and path.suffix.lower() in IMAGE_SUFFIXES:
            return path
    return None


def run_stage(image_id: str, stage: str) -> dict:
    if stage == "preprocess":
        return _run_preprocess(image_id)
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
