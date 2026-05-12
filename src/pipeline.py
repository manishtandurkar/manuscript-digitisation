"""Stage 7 — Pipeline Orchestration.

End-to-end processing for single images and batch directories.  Each stage
runs sequentially per image; batch mode uses ``multiprocessing.Pool`` for
parallelism across images.

Usage (Python)::

    from src.pipeline import process_single, process_batch

    record = process_single(
        "data/raw/tamil_stone/IMG_3941.jpg",
        artefact_meta={"type": "stone_inscription", "location": {"site": "Mamallapuram"}},
        script="tamil",
    )

    records = process_batch(
        "data/raw/tamil_stone",
        meta_csv="data/batch_meta.csv",
        workers=4,
    )

Usage (CLI)::

    python -m src.pipeline single data/raw/tamil_stone/IMG_3941.jpg
    python -m src.pipeline batch data/raw/tamil_stone --workers 2
"""
from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger("pipeline")

_PROJECT_ROOT  = Path(__file__).resolve().parents[1]
_PREPROCESSED  = _PROJECT_ROOT / "data" / "preprocessed"
_ENHANCED      = _PROJECT_ROOT / "data" / "enhanced"
_BINARISED     = _PROJECT_ROOT / "data" / "binarised"
_TRANSCRIPTIONS = _PROJECT_ROOT / "data" / "transcriptions"
_RECORDS        = _PROJECT_ROOT / "data" / "records"


# ─── Single-image pipeline ────────────────────────────────────────────────────

def process_single(
    image_path: str,
    artefact_meta: Optional[dict] = None,
    script: str = "auto",
    use_dstretch: bool = False,
    binarise_method: str = "sauvola",
    save_record: bool = True,
    export_pdf: bool = False,
) -> dict:
    """Run the full pipeline on a single image.

    Parameters
    ----------
    image_path:
        Path to the raw source image.
    artefact_meta:
        Optional artefact metadata dict (type, location, period, etc.).
    script:
        Script for OCR — ``'auto'`` detects automatically.
    use_dstretch:
        Pass ``True`` to use DStretch in the enhancement stage (for cave paintings).
    binarise_method:
        Binarisation method: ``'sauvola'`` | ``'otsu'`` | ``'adaptive'``.
    save_record:
        If ``True``, write the assembled record to ``data/records/``.
    export_pdf:
        If ``True``, export a PDF to ``outputs/exports/``.

    Returns
    -------
    dict
        Assembled research record.
    """
    from src.preprocess import preprocess
    from src.enhance import enhance
    from src.binarise import binarise
    from src.ocr import transcribe, build_output_path as ocr_output_path
    from src.metrics import full_quality_report
    from src.record import assemble_record
    from src.record import save_record as _save_record
    from src.record import export_pdf as _export_pdf

    import cv2

    src = Path(image_path)
    if not src.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    meta = artefact_meta or {}
    log: list[dict] = []

    def _stage(name: str, fn, *args, **kwargs):
        t0 = time.monotonic()
        try:
            result = fn(*args, **kwargs)
            log.append({"stage": name, "status": "success", "duration_s": round(time.monotonic() - t0, 2)})
            return result
        except Exception as exc:
            duration = round(time.monotonic() - t0, 2)
            LOGGER.error("Stage '%s' failed: %s", name, exc)
            log.append({"stage": name, "status": "failed", "error": str(exc), "duration_s": duration})
            return None

    stem = src.stem

    # ── Stage 1: Preprocess ───────────────────────────────────────────────────
    LOGGER.info("[1/5] Preprocessing %s", src.name)
    _PREPROCESSED.mkdir(parents=True, exist_ok=True)
    pre_out = _PREPROCESSED / f"{stem}_preprocessed.jpg"
    preprocessed_img = _stage("preprocess", preprocess, str(src), str(pre_out))
    pre_path = pre_out if pre_out.exists() else src

    # ── Stage 2: Enhance ──────────────────────────────────────────────────────
    LOGGER.info("[2/5] Enhancing %s", src.name)
    _ENHANCED.mkdir(parents=True, exist_ok=True)
    enh_suffix = "dstretch" if use_dstretch else "superres"
    enh_out = _ENHANCED / f"{stem}_enhanced_{enh_suffix}.jpg"
    enhanced_img = _stage(
        "enhance", enhance, str(pre_path), str(enh_out),
        use_dstretch=use_dstretch
    )
    enh_path = enh_out if enh_out.exists() else pre_path

    # ── Stage 3: Binarise ─────────────────────────────────────────────────────
    LOGGER.info("[3/5] Binarising %s", src.name)
    _BINARISED.mkdir(parents=True, exist_ok=True)
    bin_out = _BINARISED / f"{stem}_binarised.png"
    _stage("binarise", binarise, str(enh_path), str(bin_out), method=binarise_method)
    bin_path = bin_out if bin_out.exists() else enh_path

    # ── Stage 4: OCR ──────────────────────────────────────────────────────────
    LOGGER.info("[4/5] Transcribing %s", src.name)
    _TRANSCRIPTIONS.mkdir(parents=True, exist_ok=True)
    tx_out = _TRANSCRIPTIONS / f"{stem}_transcription.json"
    transcription = _stage("ocr", transcribe, str(bin_path), script, str(tx_out))
    if transcription is None:
        transcription = {"script": "unknown", "text": "", "lines": [], "overall_confidence": 0.0,
                         "confidence_status": "uncertain", "engine_used": "failed", "uncertain_regions": []}

    # ── Quality metrics ───────────────────────────────────────────────────────
    quality: dict = {}
    orig_img = cv2.imread(str(src))
    enh_img  = cv2.imread(str(enh_path))
    if orig_img is not None and enh_img is not None:
        try:
            quality = full_quality_report(orig_img, enh_img)
        except Exception as exc:
            LOGGER.warning("Quality metrics failed: %s", exc)

    # ── Stage 5: Record assembly ──────────────────────────────────────────────
    LOGGER.info("[5/5] Assembling record for %s", src.name)
    record = assemble_record(
        image_path=str(src),
        artefact_meta=meta,
        transcription=transcription,
        translation=None,
        processing_log=log,
        quality_report=quality,
    )

    if save_record:
        _save_record(record, str(_RECORDS))

    if export_pdf:
        try:
            _export_pdf(record)
        except ImportError:
            LOGGER.warning("fpdf2 not installed — skipping PDF export")
        except Exception as exc:
            LOGGER.warning("PDF export failed: %s", exc)

    LOGGER.info(
        "Pipeline complete for %s — record_id=%s confidence=%.2f",
        src.name,
        record.get("record_id"),
        transcription.get("overall_confidence", 0.0),
    )
    return record


# ─── Batch pipeline ───────────────────────────────────────────────────────────

def _process_worker(args: tuple) -> dict:
    """Top-level function for multiprocessing (must be picklable)."""
    image_path, meta, script, use_dstretch, binarise_method = args
    try:
        return process_single(
            image_path,
            artefact_meta=meta,
            script=script,
            use_dstretch=use_dstretch,
            binarise_method=binarise_method,
        )
    except Exception as exc:
        LOGGER.error("Worker failed for %s: %s", image_path, exc)
        return {"error": str(exc), "image_path": image_path}


def _load_meta_csv(csv_path: str) -> dict[str, dict]:
    """Load batch_meta.csv into a filename→meta dict."""
    meta_map: dict[str, dict] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            filename = row.pop("filename", None)
            if filename:
                meta_map[filename] = dict(row)
    return meta_map


def process_batch(
    input_dir: str,
    pattern: str = "*.jpg",
    meta_csv: Optional[str] = None,
    script: str = "auto",
    use_dstretch: bool = False,
    binarise_method: str = "sauvola",
    workers: int = 4,
) -> list[dict]:
    """Process all images in *input_dir* in parallel.

    Parameters
    ----------
    input_dir:
        Directory containing raw images.
    pattern:
        Glob pattern for selecting images (default: ``*.jpg``).
    meta_csv:
        Optional CSV path with per-image artefact metadata.  If provided,
        must have a ``filename`` column.
    script:
        Script for OCR — ``'auto'`` detects each image independently.
    workers:
        Number of parallel worker processes.

    Returns
    -------
    list[dict]
        List of assembled records (one per image).  Failed images produce
        error dicts instead of full records.
    """
    import multiprocessing

    in_dir = Path(input_dir)
    paths = sorted(in_dir.glob(pattern))
    if not paths:
        LOGGER.warning("No images found matching '%s' in %s", pattern, in_dir)
        return []

    meta_map = _load_meta_csv(meta_csv) if meta_csv else {}

    work_items = [
        (str(p), meta_map.get(p.name, {}), script, use_dstretch, binarise_method)
        for p in paths
    ]

    LOGGER.info("Batch: %d images, %d workers", len(work_items), workers)

    if workers <= 1:
        return [_process_worker(item) for item in work_items]

    with multiprocessing.Pool(processes=workers) as pool:
        records = pool.map(_process_worker, work_items)

    success = sum(1 for r in records if "error" not in r)
    LOGGER.info("Batch complete: %d/%d succeeded", success, len(records))
    return records


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inscription Digitisation Pipeline")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = p.add_subparsers(dest="mode", required=True)

    s = sub.add_parser("single", help="Process a single image end-to-end")
    s.add_argument("image", help="Path to raw image")
    s.add_argument("--script", default="auto")
    s.add_argument("--method", default="sauvola", dest="binarise_method")
    s.add_argument("--dstretch", action="store_true")
    s.add_argument("--pdf", action="store_true", help="Export PDF")
    s.add_argument("--type",     default=None, dest="artefact_type")
    s.add_argument("--location", default=None)
    s.add_argument("--period",   default=None)

    b = sub.add_parser("batch", help="Process all images in a directory")
    b.add_argument("input_dir")
    b.add_argument("--pattern",  default="*.jpg")
    b.add_argument("--meta-csv", default=None)
    b.add_argument("--script",   default="auto")
    b.add_argument("--method",   default="sauvola", dest="binarise_method")
    b.add_argument("--dstretch", action="store_true")
    b.add_argument("--workers",  type=int, default=4)
    b.add_argument("--pdf", action="store_true")

    return p


def main() -> None:
    args = _make_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(levelname)s %(name)s — %(message)s")

    if args.mode == "single":
        meta = {}
        if args.artefact_type:
            meta["type"] = args.artefact_type
        if args.location:
            meta["location"] = {"site": args.location}
        if args.period:
            meta["period_estimate"] = args.period

        record = process_single(
            args.image,
            artefact_meta=meta,
            script=args.script,
            use_dstretch=args.dstretch,
            binarise_method=args.binarise_method,
            export_pdf=args.pdf,
        )
        print(f"\nRecord: {record['record_id']}")
        print(f"Script: {record['transcription']['script']}")
        print(f"Confidence: {record['transcription']['overall_confidence']:.0%} ({record['transcription']['confidence_status']})")
        print(f"Text preview: {record['transcription']['text'][:120]}")

    else:
        records = process_batch(
            args.input_dir,
            pattern=args.pattern,
            meta_csv=args.meta_csv,
            script=args.script,
            binarise_method=args.binarise_method,
            use_dstretch=args.dstretch,
            workers=args.workers,
        )
        print(f"\nProcessed {len(records)} images.")
        for r in records:
            rid  = r.get("record_id", "ERROR")
            conf = r.get("transcription", {}).get("overall_confidence", 0)
            print(f"  {rid}: confidence={conf:.0%}")


if __name__ == "__main__":
    main()
