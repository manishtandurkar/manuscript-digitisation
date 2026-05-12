"""Quality evaluation metrics for processed inscription images (ECE responsibility).

Computes PSNR, SSIM, CNR, and sharpness scores.  All functions accept BGR or
greyscale numpy arrays and return float values.  ``full_quality_report`` is
the primary entry point and is called from ``src/pipeline.py`` to attach a
quality score to every processed record.

Minimum acceptable thresholds per project spec:
  - PSNR  >= 30 dB
  - SSIM  >= 0.85
  - CNR   >= 1.5  (project-defined; higher = better text/background separation)
  - Sharpness (Laplacian variance) — no hard threshold; logged for trending
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

LOGGER = logging.getLogger("metrics")

# ─── Thresholds (per project spec) ───────────────────────────────────────────

THRESHOLD_PSNR      = 30.0
THRESHOLD_SSIM      = 0.85
THRESHOLD_CNR       = 1.5


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _to_grey(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()


def _ensure_same_shape(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    return a, b


# ─── Individual metrics ───────────────────────────────────────────────────────

def compute_psnr(original: np.ndarray, enhanced: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio between *original* and *enhanced*.

    Higher is better.  Target >= 30 dB per project spec.
    Returns ``float('inf')`` when the images are identical.
    """
    from skimage.metrics import peak_signal_noise_ratio

    orig_g, enh_g = _ensure_same_shape(_to_grey(original), _to_grey(enhanced))
    try:
        return float(peak_signal_noise_ratio(orig_g, enh_g, data_range=255))
    except Exception as exc:
        LOGGER.warning("PSNR computation failed: %s", exc)
        return 0.0


def compute_ssim(original: np.ndarray, enhanced: np.ndarray) -> float:
    """Structural Similarity Index between *original* and *enhanced*.

    Range 0–1.  Target >= 0.85.
    """
    from skimage.metrics import structural_similarity

    orig_g, enh_g = _ensure_same_shape(_to_grey(original), _to_grey(enhanced))
    try:
        return float(structural_similarity(orig_g, enh_g, data_range=255))
    except Exception as exc:
        LOGGER.warning("SSIM computation failed: %s", exc)
        return 0.0


def compute_cnr(img: np.ndarray, text_mask: Optional[np.ndarray] = None) -> float:
    """Contrast-to-Noise Ratio between text (foreground) and background.

    If *text_mask* is provided (binary uint8, 255 = text pixels), it is used
    directly.  Otherwise, Otsu binarisation derives a mask automatically.

    CNR = |μ_text − μ_bg| / σ_bg  where σ_bg is the background std dev.
    """
    grey = _to_grey(img)

    if text_mask is None:
        _, text_mask = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    text_mask_bool = text_mask.astype(bool)
    bg_mask_bool   = ~text_mask_bool

    text_pixels = grey[text_mask_bool].astype(float)
    bg_pixels   = grey[bg_mask_bool].astype(float)

    if len(text_pixels) == 0 or len(bg_pixels) == 0:
        return 0.0

    sigma_bg = float(np.std(bg_pixels))
    if sigma_bg < 1e-6:
        return float("inf")

    return float(abs(np.mean(text_pixels) - np.mean(bg_pixels)) / sigma_bg)


def compute_sharpness(img: np.ndarray) -> float:
    """Sharpness proxy using Laplacian variance.

    Higher values indicate sharper edges.  No hard threshold — useful for
    trending (before/after enhancement).
    """
    grey = _to_grey(img)
    lap = cv2.Laplacian(grey, cv2.CV_64F)
    return float(lap.var())


# ─── Consolidated report ─────────────────────────────────────────────────────

def full_quality_report(
    original: np.ndarray,
    enhanced: np.ndarray,
    text_mask: Optional[np.ndarray] = None,
) -> dict:
    """Compute all quality metrics and return a consolidated report.

    Parameters
    ----------
    original:
        The raw / preprocessed source image (before enhancement).
    enhanced:
        The processed output image (after enhancement / binarisation).
    text_mask:
        Optional binary mask where 255 = text region.  If ``None``, a mask
        is derived automatically via Otsu thresholding.

    Returns
    -------
    dict
        Keys: ``psnr``, ``ssim``, ``cnr``, ``sharpness_original``,
        ``sharpness_enhanced``, ``sharpness_delta``, ``passes_thresholds``,
        ``threshold_check``.
    """
    psnr  = compute_psnr(original, enhanced)
    ssim  = compute_ssim(original, enhanced)
    cnr   = compute_cnr(enhanced, text_mask)
    sharp_orig = compute_sharpness(original)
    sharp_enh  = compute_sharpness(enhanced)

    passes = {
        "psnr": psnr >= THRESHOLD_PSNR,
        "ssim": ssim >= THRESHOLD_SSIM,
        "cnr":  cnr  >= THRESHOLD_CNR,
    }

    report = {
        "psnr":               round(psnr, 2),
        "ssim":               round(ssim, 4),
        "cnr":                round(cnr, 2) if cnr != float("inf") else None,
        "sharpness_original": round(sharp_orig, 2),
        "sharpness_enhanced": round(sharp_enh, 2),
        "sharpness_delta":    round(sharp_enh - sharp_orig, 2),
        "passes_thresholds":  all(passes.values()),
        "threshold_check":    passes,
    }

    LOGGER.info(
        "Quality report: PSNR=%.1f dB, SSIM=%.3f, CNR=%.2f, sharpness Δ=%.1f — %s",
        psnr, ssim, cnr if cnr != float("inf") else 999,
        report["sharpness_delta"],
        "PASS" if report["passes_thresholds"] else "NEEDS_REVIEW",
    )

    return report
