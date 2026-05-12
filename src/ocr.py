"""Stage 4 — OCR & Transcription.

Runs Tesseract and/or EasyOCR on a binarised inscription image and returns a
structured transcription dict with per-line confidence scores and bounding boxes.
Both OCR engines degrade gracefully if not installed.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

LOGGER = logging.getLogger("ocr")

# ─── Script configuration ────────────────────────────────────────────────────

SCRIPT_CONFIG: dict[str, dict[str, Any]] = {
    "tamil":      {"tesseract_lang": "tam", "easyocr_lang": ["ta"]},
    "sanskrit":   {"tesseract_lang": "san", "easyocr_lang": ["hi"]},
    "kannada":    {"tesseract_lang": "kan", "easyocr_lang": ["kn"]},
    "telugu":     {"tesseract_lang": "tel", "easyocr_lang": ["te"]},
    "malayalam":  {"tesseract_lang": "mal", "easyocr_lang": ["ml"]},
    "devanagari": {"tesseract_lang": "hin", "easyocr_lang": ["hi"]},
    "brahmi":     {"tesseract_lang": None,  "easyocr_lang": None},
    "grantha":    {"tesseract_lang": None,  "easyocr_lang": None},
}

_CONFIDENCE_VERIFIED = 0.85
_CONFIDENCE_REVIEW   = 0.60

# Tesseract PSM/OEM per spec: LSTM engine, uniform block of text
_TESS_CONFIG = "--oem 1 --psm 6"

# ─── Optional import guards ──────────────────────────────────────────────────

try:
    import pytesseract
    _TESS_AVAILABLE = True
except ImportError:
    _TESS_AVAILABLE = False
    LOGGER.warning("pytesseract not installed — Tesseract OCR unavailable")

try:
    import easyocr as _easyocr_module
    _EASY_AVAILABLE = True
except ImportError:
    _EASY_AVAILABLE = False
    LOGGER.warning("easyocr not installed — EasyOCR unavailable")

_EASYOCR_READER_CACHE: dict[str, Any] = {}


# ─── Script detection ────────────────────────────────────────────────────────

def detect_script(img: np.ndarray) -> str:
    """Heuristic script detection based on connected-component properties.

    Returns a script name from SCRIPT_CONFIG or 'unknown'.  Tamil is the
    default fallback since it is the project's primary test script.
    """
    if img is None or img.size == 0:
        return "tamil"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

    # Binarise if not already binary
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return "tamil"

    # Remove background label (label 0)
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    component_h     = stats[1:, cv2.CC_STAT_HEIGHT]
    component_w     = stats[1:, cv2.CC_STAT_WIDTH]

    if len(component_areas) == 0:
        return "tamil"

    median_area   = float(np.median(component_areas))
    median_aspect = float(np.median(component_w / (component_h + 1e-6)))

    # Devanagari / Hindi — components have a distinctive top bar (śirorekha).
    # They tend to be taller relative to width (aspect ratio < 1).
    if median_aspect < 0.7 and median_area > 100:
        return "devanagari"

    # Tamil characters are compact and roughly square to slightly wide.
    # Default to Tamil for South-Indian square-ish glyphs.
    return "tamil"


# ─── Tesseract OCR ───────────────────────────────────────────────────────────

def ocr_tesseract(img: np.ndarray, lang: str) -> dict:
    """Run Tesseract OCR on *img* using the given language code.

    Returns::

        {
            "text": str,
            "confidence": float,          # mean word confidence 0–1
            "word_boxes": [               # one entry per word
                {"text": str, "confidence": float, "box": [x, y, w, h]}
            ]
        }

    Returns an empty result if pytesseract is unavailable or *lang* is None.
    """
    empty: dict = {"text": "", "confidence": 0.0, "word_boxes": []}

    if not _TESS_AVAILABLE or not lang:
        return empty

    try:
        # data["conf"] values: -1 = non-text, 0–100 = confidence
        data = pytesseract.image_to_data(
            img,
            lang=lang,
            config=_TESS_CONFIG,
            output_type=pytesseract.Output.DICT,
        )
    except Exception as exc:
        LOGGER.warning("Tesseract failed: %s", exc)
        return empty

    word_boxes: list[dict] = []
    valid_confs: list[float] = []

    for i, conf in enumerate(data["conf"]):
        if conf == -1:
            continue
        word_text = (data["text"][i] or "").strip()
        if not word_text:
            continue
        conf_norm = conf / 100.0
        word_boxes.append({
            "text": word_text,
            "confidence": round(conf_norm, 4),
            "box": [data["left"][i], data["top"][i], data["width"][i], data["height"][i]],
        })
        valid_confs.append(conf_norm)

    full_text = " ".join(wb["text"] for wb in word_boxes)
    mean_conf = float(np.mean(valid_confs)) if valid_confs else 0.0

    return {"text": full_text, "confidence": round(mean_conf, 4), "word_boxes": word_boxes}


# ─── EasyOCR ─────────────────────────────────────────────────────────────────

def _get_easyocr_reader(langs: list[str]) -> Any | None:
    """Return a cached EasyOCR Reader for the given language list."""
    if not _EASY_AVAILABLE:
        return None
    key = ",".join(sorted(langs))
    if key not in _EASYOCR_READER_CACHE:
        try:
            _EASYOCR_READER_CACHE[key] = _easyocr_module.Reader(
                langs, gpu=False, verbose=False
            )
        except Exception as exc:
            LOGGER.warning("EasyOCR Reader init failed: %s", exc)
            return None
    return _EASYOCR_READER_CACHE[key]


def ocr_easyocr(img: np.ndarray, langs: list[str]) -> dict:
    """Run EasyOCR on *img*.

    Returns the same schema as :func:`ocr_tesseract`.
    """
    empty: dict = {"text": "", "confidence": 0.0, "word_boxes": []}

    if not _EASY_AVAILABLE or not langs:
        return empty

    reader = _get_easyocr_reader(langs)
    if reader is None:
        return empty

    try:
        results = reader.readtext(img, detail=1)
    except Exception as exc:
        LOGGER.warning("EasyOCR readtext failed: %s", exc)
        return empty

    word_boxes: list[dict] = []
    valid_confs: list[float] = []

    for bbox, text, conf in results:
        text = (text or "").strip()
        if not text:
            continue
        # bbox is [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        x, y = int(min(xs)), int(min(ys))
        w = int(max(xs) - min(xs))
        h = int(max(ys) - min(ys))
        word_boxes.append({
            "text": text,
            "confidence": round(float(conf), 4),
            "box": [x, y, w, h],
        })
        valid_confs.append(float(conf))

    full_text = " ".join(wb["text"] for wb in word_boxes)
    mean_conf = float(np.mean(valid_confs)) if valid_confs else 0.0

    return {"text": full_text, "confidence": round(mean_conf, 4), "word_boxes": word_boxes}


# ─── Ensemble ────────────────────────────────────────────────────────────────

def ocr_ensemble(img: np.ndarray, script: str) -> dict:
    """Run both engines and merge by confidence.

    Whichever engine returns a higher mean confidence wins for the overall
    text field.  Word boxes from both engines are combined (deduplicated by
    position) to give a richer set of bounding boxes for uncertain-region
    detection.
    """
    cfg = SCRIPT_CONFIG.get(script, SCRIPT_CONFIG["tamil"])
    tess_lang  = cfg["tesseract_lang"]
    easy_langs = cfg["easyocr_lang"]

    tess = ocr_tesseract(img, tess_lang)
    easy = ocr_easyocr(img, easy_langs or [])

    # Pick higher-confidence result as primary text
    if tess["confidence"] >= easy["confidence"]:
        primary, secondary = tess, easy
        engine_used = "tesseract"
        if easy["confidence"] > 0:
            engine_used = "tesseract+easyocr ensemble"
    else:
        primary, secondary = easy, tess
        engine_used = "easyocr"
        if tess["confidence"] > 0:
            engine_used = "tesseract+easyocr ensemble"

    # Merge word boxes: primary first, then secondary boxes that don't overlap
    merged_boxes = list(primary["word_boxes"])
    primary_rects = {(wb["box"][0], wb["box"][1]) for wb in primary["word_boxes"]}
    for wb in secondary["word_boxes"]:
        key = (wb["box"][0], wb["box"][1])
        if key not in primary_rects:
            merged_boxes.append(wb)

    return {
        "text": primary["text"],
        "confidence": primary["confidence"],
        "word_boxes": merged_boxes,
        "engine_used": engine_used,
        "tess_confidence": tess["confidence"],
        "easy_confidence": easy["confidence"],
    }


# ─── Line extraction ─────────────────────────────────────────────────────────

def _group_words_into_lines(word_boxes: list[dict]) -> list[dict]:
    """Group word boxes into lines by proximity on the y-axis.

    Returns a list of line dicts sorted by y position::

        [{"line_number": int, "text": str, "confidence": float,
          "bounding_box": [x, y, w, h], "uncertain": bool}]
    """
    if not word_boxes:
        return []

    # Sort by top-y then left-x
    sorted_words = sorted(word_boxes, key=lambda wb: (wb["box"][1], wb["box"][0]))

    lines: list[list[dict]] = []
    current_line: list[dict] = [sorted_words[0]]
    current_y = sorted_words[0]["box"][1]
    current_h = sorted_words[0]["box"][3]

    for wb in sorted_words[1:]:
        word_y = wb["box"][1]
        # Words on the same line if their top-y is within half the line height
        if abs(word_y - current_y) <= max(current_h * 0.6, 8):
            current_line.append(wb)
        else:
            lines.append(current_line)
            current_line = [wb]
            current_y = word_y
            current_h = wb["box"][3]
    lines.append(current_line)

    result: list[dict] = []
    for idx, line_words in enumerate(lines, start=1):
        line_words_sorted = sorted(line_words, key=lambda wb: wb["box"][0])
        text = " ".join(wb["text"] for wb in line_words_sorted)
        confs = [wb["confidence"] for wb in line_words_sorted]
        mean_conf = float(np.mean(confs)) if confs else 0.0

        xs = [wb["box"][0] for wb in line_words_sorted]
        ys = [wb["box"][1] for wb in line_words_sorted]
        x2s = [wb["box"][0] + wb["box"][2] for wb in line_words_sorted]
        y2s = [wb["box"][1] + wb["box"][3] for wb in line_words_sorted]
        bx = min(xs); by = min(ys)
        bw = max(x2s) - bx; bh = max(y2s) - by

        result.append({
            "line_number": idx,
            "text": text,
            "confidence": round(mean_conf, 4),
            "bounding_box": [bx, by, bw, bh],
            "uncertain": mean_conf < _CONFIDENCE_REVIEW,
        })

    return result


# ─── Main transcribe function ────────────────────────────────────────────────

def transcribe(
    img_path: str,
    script: str = "auto",
    output_path: str | None = None,
) -> dict:
    """Full OCR & transcription pipeline.

    Parameters
    ----------
    img_path:
        Path to the binarised (or enhanced) image to process.
    script:
        Script name (``'auto'`` triggers :func:`detect_script`).
    output_path:
        If given, write the transcription JSON to this path.

    Returns
    -------
    dict
        Structured transcription following the project schema::

            {
                "script": str,
                "text": str,
                "lines": [...],
                "overall_confidence": float,
                "engine_used": str,
                "uncertain_regions": [[x1, y1, x2, y2]]
            }
    """
    img_path_obj = Path(img_path)
    if not img_path_obj.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = cv2.imread(str(img_path_obj))
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    LOGGER.info("Transcribing %s", img_path_obj.name)
    t0 = time.monotonic()

    # ── Script detection ──────────────────────────────────────────────────────
    if script == "auto":
        script = detect_script(img)
        LOGGER.info("Detected script: %s", script)

    # ── Handle scripts with no OCR support ───────────────────────────────────
    cfg = SCRIPT_CONFIG.get(script, SCRIPT_CONFIG["tamil"])
    if cfg["tesseract_lang"] is None and cfg["easyocr_lang"] is None:
        LOGGER.warning(
            "No OCR engine available for script '%s' — flagging for manual transcription",
            script,
        )
        result: dict = {
            "script": script,
            "text": "",
            "lines": [],
            "overall_confidence": 0.0,
            "engine_used": "none — manual transcription required",
            "uncertain_regions": [],
            "status": "manual_transcription_required",
            "duration_s": round(time.monotonic() - t0, 2),
        }
        _maybe_save(result, output_path)
        return result

    # ── Run OCR ensemble ──────────────────────────────────────────────────────
    ensemble = ocr_ensemble(img, script)
    word_boxes = ensemble["word_boxes"]
    engine_used = ensemble.get("engine_used", "unknown")

    # ── Line grouping ─────────────────────────────────────────────────────────
    lines = _group_words_into_lines(word_boxes)

    # ── Uncertain regions (confidence < threshold) ────────────────────────────
    uncertain_regions: list[list[int]] = []
    for wb in word_boxes:
        if wb["confidence"] < _CONFIDENCE_REVIEW:
            x, y, w, h = wb["box"]
            uncertain_regions.append([x, y, x + w, y + h])

    # ── Overall confidence ────────────────────────────────────────────────────
    if lines:
        overall_conf = float(np.mean([ln["confidence"] for ln in lines]))
    elif word_boxes:
        overall_conf = ensemble["confidence"]
    else:
        overall_conf = 0.0

    # ── Confidence status label ───────────────────────────────────────────────
    if overall_conf >= _CONFIDENCE_VERIFIED:
        conf_status = "verified"
    elif overall_conf >= _CONFIDENCE_REVIEW:
        conf_status = "review_needed"
    else:
        conf_status = "uncertain"

    result = {
        "script": script,
        "text": ensemble["text"],
        "lines": lines,
        "overall_confidence": round(overall_conf, 4),
        "confidence_status": conf_status,
        "engine_used": engine_used,
        "uncertain_regions": uncertain_regions,
        "duration_s": round(time.monotonic() - t0, 2),
    }

    LOGGER.info(
        "Transcription done in %.2fs — %d lines, confidence=%.2f (%s)",
        result["duration_s"],
        len(lines),
        overall_conf,
        conf_status,
    )

    _maybe_save(result, output_path)
    return result


def _maybe_save(result: dict, output_path: str | None) -> None:
    if not output_path:
        return
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Saved transcription → %s", out)


def build_output_path(input_path: str | Path, output_dir: str | Path) -> Path:
    """Return ``output_dir / {stem}_transcription.json``."""
    return Path(output_dir) / f"{Path(input_path).stem}_transcription.json"


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 4 — OCR & Transcription of inscription images"
    )
    sub = p.add_subparsers(dest="mode", required=True)

    single = sub.add_parser("single", help="Transcribe a single image")
    single.add_argument("input", help="Path to input image (binarised preferred)")
    single.add_argument("output", help="Path to output JSON file")
    single.add_argument(
        "--script",
        default="auto",
        choices=["auto"] + list(SCRIPT_CONFIG.keys()),
        help="Script name (default: auto-detect)",
    )

    batch = sub.add_parser("batch", help="Transcribe all images in a directory")
    batch.add_argument("input_dir", help="Directory containing binarised images")
    batch.add_argument("output_dir", help="Directory for output JSON files")
    batch.add_argument(
        "--pattern", default="*.png", help="Glob pattern (default: *.png)"
    )
    batch.add_argument(
        "--script",
        default="auto",
        choices=["auto"] + list(SCRIPT_CONFIG.keys()),
    )

    for parser in (single, batch):
        parser.add_argument(
            "--log-level",
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        )

    return p


def main() -> None:
    args = _make_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    if args.mode == "single":
        out = transcribe(args.input, script=args.script, output_path=args.output)
        print(f"Script: {out['script']}")
        print(f"Confidence: {out['overall_confidence']} ({out['confidence_status']})")
        print(f"Text:\n{out['text']}")
    else:
        in_dir = Path(args.input_dir)
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = sorted(in_dir.glob(args.pattern))
        LOGGER.info("Processing %d images", len(paths))
        for p in paths:
            out_path = build_output_path(p, out_dir)
            transcribe(str(p), script=args.script, output_path=str(out_path))


if __name__ == "__main__":
    main()
