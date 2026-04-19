from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENHANCED_DIR = _PROJECT_ROOT / "data" / "enhanced"
MODEL_DIR = _PROJECT_ROOT / "models" / "weights"
DEFAULT_MODEL_PATH = MODEL_DIR / "RealESRGAN_x4plus.pth"
MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

LOGGER = logging.getLogger("enhance")


def denoise(img: np.ndarray, strength: int = 10) -> np.ndarray:
    """Non-local means denoising. strength=10 mild, 20 heavy."""
    return cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)


def dstretch(img: np.ndarray, colour_space: str = "LAB") -> np.ndarray:
    """Decorrelation stretch — reveals faded pigment invisible to the eye."""
    img_float = img.astype(np.float64) / 255.0
    flat = img_float.reshape(-1, 3)
    mean = flat.mean(axis=0)
    centered = flat - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    stretch_matrix = (
        eigenvectors
        @ np.diag(1.0 / np.sqrt(eigenvalues + 1e-10))
        @ eigenvectors.T
    )
    stretched = centered @ stretch_matrix.T
    lo, hi = stretched.min(), stretched.max()
    stretched = (stretched - lo) / (hi - lo + 1e-10)
    return (stretched.reshape(img_float.shape) * 255).astype(np.uint8)


def sharpen(img: np.ndarray, amount: float = 1.5) -> np.ndarray:
    """Unsharp mask sharpening to crisp up character edges."""
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)
