from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

LOGGER = logging.getLogger("binarise")


def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _clahe(gray: np.ndarray) -> np.ndarray:
    """CLAHE equalization — normalises uneven illumination before thresholding."""
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def _sauvola_window(gray: np.ndarray) -> int:
    """Window ~1/20 of shorter dimension, clamped to [15, 51], always odd."""
    w = max(15, min(gray.shape[0], gray.shape[1]) // 20)
    return w if w % 2 == 1 else w + 1


def binarise_sauvola(img: np.ndarray, window_size: int | None = None) -> np.ndarray:
    """Sauvola local thresholding — best for uneven backgrounds (stone, palm leaf)."""
    from skimage.filters import threshold_sauvola

    gray = _clahe(_to_gray(img))
    ws = window_size if window_size is not None else _sauvola_window(gray)
    thresh = threshold_sauvola(gray, window_size=ws)
    binary = (gray > thresh).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def binarise_otsu(img: np.ndarray) -> np.ndarray:
    """Otsu global thresholding — fast, good for clean paper manuscripts."""
    gray = _clahe(_to_gray(img))
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def binarise_adaptive(img: np.ndarray) -> np.ndarray:
    """OpenCV adaptive mean thresholding — fallback for mixed quality images."""
    gray = _clahe(_to_gray(img))
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 8
    )
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def remove_noise_blobs(binary: np.ndarray, min_size: int = 50) -> np.ndarray:
    """Remove small disconnected components (dust, noise) from binary image."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    cleaned = np.zeros_like(binary)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_size:
            cleaned[labels == label] = 255
    return cleaned


def binarise(
    img_path: str,
    output_path: str,
    method: str = "sauvola",
) -> np.ndarray:
    """Binarise image. method: 'sauvola' | 'otsu' | 'adaptive'"""
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    if method == "sauvola":
        binary = binarise_sauvola(img)
    elif method == "otsu":
        binary = binarise_otsu(img)
    elif method == "adaptive":
        binary = binarise_adaptive(img)
    else:
        raise ValueError(f"Unknown method '{method}'. Use: sauvola | otsu | adaptive")

    binary = remove_noise_blobs(binary)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), binary)

    LOGGER.info("Binarised %s → %s (method=%s)", img_path, out, method)
    return binary


def build_output_path(input_path: Path, output_dir: Path) -> Path:
    return Path(output_dir) / f"{Path(input_path).stem}_binarised.png"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3: Binarise inscription images")
    sub = parser.add_subparsers(dest="mode", required=True)

    single = sub.add_parser("single", help="Process one image")
    single.add_argument("input", help="Input image path")
    single.add_argument("output", help="Output PNG path")
    single.add_argument(
        "--method", choices=["sauvola", "otsu", "adaptive"], default="sauvola"
    )

    batch = sub.add_parser("batch", help="Process a directory")
    batch.add_argument("input_dir", help="Directory of images")
    batch.add_argument("output_dir", help="Directory for output PNGs")
    batch.add_argument(
        "--method", choices=["sauvola", "otsu", "adaptive"], default="sauvola"
    )
    batch.add_argument("--pattern", default="*.jpg", help="Glob pattern")

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.mode == "single":
        binarise(args.input, args.output, method=args.method)
    else:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        images = list(input_dir.glob(args.pattern))
        LOGGER.info("Found %d images in %s", len(images), input_dir)
        for img_path in images:
            out_path = build_output_path(img_path, output_dir)
            try:
                binarise(str(img_path), str(out_path), method=args.method)
            except Exception as exc:
                LOGGER.error("Failed %s: %s", img_path, exc)


if __name__ == "__main__":
    main()
