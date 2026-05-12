"""Stage 6 — Record Assembly.

Bundles all pipeline outputs (images, transcription, quality metrics, metadata)
into a structured JSON research record and optionally exports a PDF.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

LOGGER = logging.getLogger("record")

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_RECORDS_DIR  = _PROJECT_ROOT / "data" / "records"

# ─── Record ID generation ─────────────────────────────────────────────────────

def generate_record_id(year: Optional[int] = None) -> str:
    """Auto-generate sequential record ID in the format ``INS-YYYY-NNNN``.

    Scans ``data/records/`` to find the highest existing sequence number and
    increments it.  Thread-safe within a single process via sequential file
    listing (no file locking needed for the current single-process pipeline).
    """
    yr = year or datetime.now(timezone.utc).year
    pattern = re.compile(rf"INS-{yr}-(\d{{4}})\.json")

    existing: list[int] = []
    if _RECORDS_DIR.exists():
        for f in _RECORDS_DIR.iterdir():
            m = pattern.match(f.name)
            if m:
                existing.append(int(m.group(1)))

    next_seq = (max(existing) + 1) if existing else 1
    return f"INS-{yr}-{next_seq:04d}"


# ─── Record assembly ──────────────────────────────────────────────────────────

def assemble_record(
    image_path: str,
    artefact_meta: dict,
    transcription: dict,
    translation: Optional[dict] = None,
    processing_log: Optional[list] = None,
    quality_report: Optional[dict] = None,
) -> dict:
    """Assemble a full research record dict from all pipeline outputs.

    Parameters
    ----------
    image_path:
        Path to the original (raw) image.
    artefact_meta:
        Dict with keys such as ``type``, ``material``, ``period_estimate``,
        ``dynasty``, ``location``, ``condition``, ``collection``,
        ``accession_number``.  All fields are optional.
    transcription:
        Output dict from ``src.ocr.transcribe``.
    translation:
        Output dict from ``src.translate.translate`` (Phase 2).  If ``None``,
        the translation block is filled with Phase-2-pending placeholders.
    processing_log:
        List of stage-level dicts::

            [{"stage": str, "status": str, "duration_s": float}]

    quality_report:
        Output dict from ``src.metrics.full_quality_report``.

    Returns
    -------
    dict
        Complete record following the project JSON schema.
    """
    img = Path(image_path)
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    record_id = generate_record_id()

    # ── Image paths (relative to project root) ────────────────────────────────
    stem = img.stem
    images_block: dict[str, Any] = {
        "original": str(img),
        "enhanced": str(_PROJECT_ROOT / "data" / "enhanced" / f"{stem}_enhanced_superres.jpg"),
        "binarised": str(_PROJECT_ROOT / "data" / "binarised" / f"{stem}_binarised.png"),
        "thumbnail": str(_PROJECT_ROOT / "data" / "thumbnails" / f"{stem}_thumb.jpg"),
        "enhancement_method": "RealESRGAN_x4plus + OpenCV denoising",
        "processed_at": now,
    }

    # ── Artefact block ────────────────────────────────────────────────────────
    artefact_block: dict[str, Any] = {
        "type": artefact_meta.get("type", "unknown"),
        "material": artefact_meta.get("material"),
        "period_estimate": artefact_meta.get("period_estimate"),
        "dynasty": artefact_meta.get("dynasty"),
        "location": artefact_meta.get("location", {}),
        "dimensions_cm": artefact_meta.get("dimensions_cm"),
        "condition": artefact_meta.get("condition", "unknown"),
        "collection": artefact_meta.get("collection"),
        "accession_number": artefact_meta.get("accession_number"),
    }

    # ── Transcription block ───────────────────────────────────────────────────
    transcription_block: dict[str, Any] = {
        "script": transcription.get("script", "unknown"),
        "language": _script_to_language(transcription.get("script", "")),
        "text": transcription.get("text", ""),
        "lines": transcription.get("lines", []),
        "overall_confidence": transcription.get("overall_confidence", 0.0),
        "confidence_status": transcription.get("confidence_status", "uncertain"),
        "uncertain_segments": [
            ln["text"] for ln in transcription.get("lines", []) if ln.get("uncertain")
        ],
        "ocr_engine": transcription.get("engine_used", "unknown"),
    }

    # ── Translation block (Phase 2 placeholder) ───────────────────────────────
    if translation:
        translation_block = translation
    else:
        translation_block = {
            "english": None,
            "modern_source_language": None,
            "confidence": None,
            "method": None,
            "notes": [],
            "status": "phase_2_pending",
        }

    # ── Quality report ────────────────────────────────────────────────────────
    quality_block = quality_report or {}

    # ── Citation ──────────────────────────────────────────────────────────────
    location_str = _location_string(artefact_meta.get("location", {}))
    period_str = artefact_meta.get("period_estimate", "date unknown")
    citation_block: dict[str, Any] = {
        "suggested_cite": (
            f"Inscription {record_id}. {location_str}. {period_str}. "
            f"Processed by Inscription Digitisation Project, {datetime.now(timezone.utc).strftime('%B %Y')}."
        ),
        "doi": None,
        "licence": "CC BY 4.0",
    }

    # ── Processing log ────────────────────────────────────────────────────────
    log = processing_log or []

    # ── Determine overall status ──────────────────────────────────────────────
    conf = transcription.get("overall_confidence", 0.0)
    status = "verified" if conf >= 0.85 else "review" if conf >= 0.60 else "draft"

    return {
        "record_id": record_id,
        "created_at": now,
        "status": status,
        "artefact": artefact_block,
        "images": images_block,
        "transcription": transcription_block,
        "translation": translation_block,
        "quality": quality_block,
        "citation": citation_block,
        "processing_log": log,
    }


# ─── Save record ──────────────────────────────────────────────────────────────

def save_record(record: dict, output_dir: Optional[str] = None) -> str:
    """Serialise *record* to JSON and write to *output_dir*.

    Returns the path of the saved file as a string.
    """
    out_dir = Path(output_dir) if output_dir else _RECORDS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    record_id = record.get("record_id", "INS-UNKNOWN")
    out_path = out_dir / f"{record_id}.json"

    out_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info("Saved record → %s", out_path)
    return str(out_path)


# ─── PDF export ──────────────────────────────────────────────────────────────

def export_pdf(record: dict, output_dir: Optional[str] = None) -> str:
    """Export a researcher-friendly PDF for the given record.

    Includes:
    - Artefact metadata table
    - Side-by-side original vs. enhanced image (if files exist)
    - Transcription with line-level confidence
    - Translation section (blank / Phase-2-pending label in Phase 1)
    - Citation block

    Requires ``fpdf2``.  If not installed, raises ``ImportError``.

    Returns the path of the saved PDF file.
    """
    try:
        from fpdf import FPDF
    except ImportError as exc:
        raise ImportError(
            "fpdf2 is required for PDF export. Install it with: pip install fpdf2"
        ) from exc

    out_dir = Path(output_dir) if output_dir else _PROJECT_ROOT / "outputs" / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)

    record_id = record.get("record_id", "INS-UNKNOWN")
    pdf_path = out_dir / f"{record_id}.pdf"

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, f"Inscription Research Record — {record_id}", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Created: {record.get('created_at', '')}   Status: {record.get('status', '')}", ln=True)
    pdf.ln(4)

    # ── Artefact metadata ──────────────────────────────────────────────────
    art = record.get("artefact", {})
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Artefact Metadata", ln=True)
    pdf.set_font("Helvetica", "", 10)
    meta_rows = [
        ("Type",       art.get("type", "—")),
        ("Material",   art.get("material") or "—"),
        ("Period",     art.get("period_estimate") or "—"),
        ("Dynasty",    art.get("dynasty") or "—"),
        ("Condition",  art.get("condition") or "—"),
        ("Collection", art.get("collection") or "—"),
        ("Accession",  art.get("accession_number") or "—"),
        ("Location",   _location_string(art.get("location", {}))),
    ]
    for label, value in meta_rows:
        pdf.cell(40, 6, f"{label}:", border=0)
        pdf.cell(0, 6, str(value), ln=True)
    pdf.ln(4)

    # ── Images ────────────────────────────────────────────────────────────
    images = record.get("images", {})
    orig_path  = Path(images.get("original", ""))
    enh_path   = Path(images.get("enhanced", ""))
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Images", ln=True)
    pdf.set_font("Helvetica", "", 10)

    available: list[tuple[str, Path]] = []
    for label, p in [("Original", orig_path), ("Enhanced", enh_path)]:
        if p.exists() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            available.append((label, p))

    if available:
        img_w = 85
        x_start = pdf.get_x()
        y_start = pdf.get_y()
        for idx, (label, p) in enumerate(available[:2]):
            x = x_start + idx * (img_w + 5)
            try:
                pdf.image(str(p), x=x, y=y_start + 6, w=img_w)
                pdf.set_xy(x, y_start)
                pdf.cell(img_w, 6, label, align="C")
            except Exception:
                pass
        pdf.ln(65)
    else:
        pdf.cell(0, 6, "(Images not available on disk)", ln=True)
    pdf.ln(4)

    # ── Transcription ─────────────────────────────────────────────────────
    tx = record.get("transcription", {})
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Transcription", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(40, 6, "Script:")
    pdf.cell(0, 6, f"{tx.get('script', '—')} — {tx.get('language', '—')}", ln=True)
    pdf.cell(40, 6, "Confidence:")
    conf_val = tx.get("overall_confidence", 0)
    pdf.cell(0, 6, f"{conf_val:.0%} ({tx.get('confidence_status', '—')})", ln=True)
    pdf.cell(40, 6, "Engine:")
    pdf.cell(0, 6, tx.get("ocr_engine", "—"), ln=True)
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Extracted Text:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    text = tx.get("text", "(no text extracted)")
    pdf.multi_cell(0, 6, text or "(no text extracted)")
    pdf.ln(4)

    # ── Translation ───────────────────────────────────────────────────────
    tr = record.get("translation", {})
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Translation", ln=True)
    pdf.set_font("Helvetica", "", 10)
    if tr.get("status") == "phase_2_pending":
        pdf.cell(0, 6, "[Translation — Phase 2 pending. Not yet implemented.]", ln=True)
    else:
        pdf.multi_cell(0, 6, tr.get("english") or "(no translation)")
    pdf.ln(4)

    # ── Quality ───────────────────────────────────────────────────────────
    q = record.get("quality", {})
    if q:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Quality Metrics", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for metric, label in [("psnr", "PSNR (dB)"), ("ssim", "SSIM"), ("cnr", "CNR"),
                               ("sharpness_delta", "Sharpness Δ")]:
            val = q.get(metric)
            if val is not None:
                pdf.cell(50, 6, f"{label}:")
                pdf.cell(0, 6, str(val), ln=True)
        pdf.ln(4)

    # ── Citation ──────────────────────────────────────────────────────────
    cite = record.get("citation", {})
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Citation", ln=True)
    pdf.set_font("Helvetica", "I", 10)
    pdf.multi_cell(0, 6, cite.get("suggested_cite", ""))
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Licence: {cite.get('licence', 'CC BY 4.0')}", ln=True)

    pdf.output(str(pdf_path))
    LOGGER.info("Exported PDF → %s", pdf_path)
    return str(pdf_path)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _script_to_language(script: str) -> str:
    mapping = {
        "tamil":      "Classical Tamil",
        "sanskrit":   "Sanskrit",
        "kannada":    "Kannada",
        "telugu":     "Telugu",
        "malayalam":  "Malayalam",
        "devanagari": "Hindi / Sanskrit (Devanagari)",
        "brahmi":     "Brahmi",
        "grantha":    "Grantha",
    }
    return mapping.get(script.lower(), script.title() if script else "Unknown")


def _location_string(location: dict) -> str:
    parts = [
        location.get("site"),
        location.get("district"),
        location.get("state"),
        location.get("country"),
    ]
    return ", ".join(p for p in parts if p) or "Location unknown"
