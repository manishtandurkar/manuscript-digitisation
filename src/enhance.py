from __future__ import annotations

from functools import lru_cache
import logging
import urllib.request
from pathlib import Path

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
    n = flat.shape[0]
    cov = (centered.T @ centered) / (n - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    if eigenvalues.max() < 1e-8:
        LOGGER.warning("dstretch: near-uniform image, skipping stretch")
        return img.copy()
    stretch_matrix = (
        eigenvectors
        @ np.diag(1.0 / np.sqrt(eigenvalues + 1e-10))
        @ eigenvectors.T
    )
    stretched = centered @ stretch_matrix
    lo = stretched.min(axis=0)
    hi = stretched.max(axis=0)
    stretched = (stretched - lo) / (hi - lo + 1e-10)
    return (stretched.reshape(img_float.shape) * 255).astype(np.uint8)


def sharpen(img: np.ndarray, amount: float = 1.5) -> np.ndarray:
    """Unsharp mask sharpening to crisp up character edges."""
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


def _download_weights(model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading Real-ESRGAN weights to %s …", model_path)
    urllib.request.urlretrieve(MODEL_URL, str(model_path))
    LOGGER.info("Download complete.")


def _build_upsampler(model_path: str):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64,
        num_block=23, num_grow_ch=32, scale=4,
    )
    return RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=400,
        tile_pad=10,
        pre_pad=0,
        half=False,
    )


@lru_cache(maxsize=2)
def _get_upsampler(model_path: str):
    return _build_upsampler(model_path)


def enhance_with_realesrgan(
    img: np.ndarray,
    scale: int = 2,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> np.ndarray:
    """Super-resolution via Real-ESRGAN. outscale=2 avoids over-smoothing."""
    mp = Path(model_path)
    if not mp.exists():
        _download_weights(mp)

    upsampler = _get_upsampler(str(mp.resolve()))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output_rgb, _ = upsampler.enhance(img_rgb, outscale=scale)
    return cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR).astype(np.uint8)


def build_output_path(input_path: Path, output_dir: Path) -> Path:
    """Returns output_dir / {stem}_enhanced.jpg"""
    return Path(output_dir) / f"{Path(input_path).stem}_enhanced.jpg"


def enhance(
    img_path: str,
    output_path: str,
    use_dstretch: bool = False,
) -> np.ndarray:
    """Full enhancement chain. Degrades gracefully if Real-ESRGAN unavailable."""
    from src.utils import save_image

    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    img = denoise(img)

    if use_dstretch:
        img = dstretch(img)
    else:
        try:
            img = enhance_with_realesrgan(img)
        except Exception as exc:  # ImportError when basicsr not installed; RuntimeError for GPU/model errors
            LOGGER.warning("Real-ESRGAN unavailable (%s) — skipping super-resolution.", exc)

    img = sharpen(img)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image(out, img)

    LOGGER.info("Enhanced %s → %s", img_path, out)
    return img
