from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import ImageOps
from PIL import Image as PilImage

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import save_image

LOGGER = logging.getLogger("preprocess")


def load_image(path: str) -> np.ndarray:
    """Load image as BGR numpy array, applying EXIF orientation so output is visually upright."""
    with PilImage.open(path) as pil_img:
        pil_img = ImageOps.exif_transpose(pil_img)
        rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def normalise_brightness(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def auto_white_balance(img: np.ndarray) -> np.ndarray:
    img_float = img.astype(np.float32)
    channel_means = img_float.reshape(-1, 3).mean(axis=0)
    overall_mean = float(channel_means.mean())
    scale = overall_mean / np.maximum(channel_means, 1e-6)
    balanced = img_float * scale.reshape(1, 1, 3)
    return np.clip(balanced, 0, 255).astype(np.uint8)


def _crop_borders_with_metadata(img: np.ndarray, threshold: int = 10) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = ((gray > threshold) & (gray < 255 - threshold)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    points = cv2.findNonZero(mask)

    if points is None:
        height, width = img.shape[:2]
        return img.copy(), (0, 0, width, height)

    x, y, w, h = cv2.boundingRect(points)
    height, width = img.shape[:2]

    if w < width * 0.25 or h < height * 0.25:
        return img.copy(), (0, 0, width, height)

    cropped = img[y : y + h, x : x + w]
    return cropped, (x, y, w, h)


def crop_borders(img: np.ndarray, threshold: int = 10) -> np.ndarray:
    """Remove blank/dark border margins from scans."""
    cropped, _ = _crop_borders_with_metadata(img, threshold=threshold)
    return cropped


def preprocess(img_path: str, output_path: str) -> np.ndarray:
    """Run full preprocessing chain and save result as JPEG."""
    input_path = Path(img_path)
    original = load_image(str(input_path))
    before_height, before_width = original.shape[:2]

    processed = normalise_brightness(original)
    processed = auto_white_balance(processed)
    processed, crop_box = _crop_borders_with_metadata(processed)

    out = Path(output_path)
    save_image(out, processed)

    after_height, after_width = processed.shape[:2]
    LOGGER.info(
        "Preprocessed %s | before=%sx%s after=%sx%s crop=%s output=%s",
        input_path.name,
        before_width,
        before_height,
        after_width,
        after_height,
        crop_box,
        out,
    )
    return processed


def build_output_path(input_path: str | Path, output_dir: str | Path) -> Path:
    """Build the output JPEG path for a preprocessed image."""
    source = Path(input_path)
    target_dir = Path(output_dir)
    return target_dir / f"{source.stem}_preprocessed.jpg"


def process_directory(
    input_dir: str,
    output_dir: str,
    pattern: str = "*.jpg",
) -> list[Path]:
    """Preprocess every matching image in a directory."""
    source_dir = Path(input_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    input_paths = sorted(source_dir.glob(pattern))
    if not input_paths:
        raise FileNotFoundError(f"No images matching {pattern!r} found in {input_dir}")

    output_paths: list[Path] = []
    total = len(input_paths)
    for index, input_path in enumerate(input_paths, start=1):
        output_path = build_output_path(input_path, output_dir)
        LOGGER.info("Batch preprocessing %s/%s: %s", index, total, input_path.name)
        preprocess(str(input_path), str(output_path))
        output_paths.append(output_path)

    return output_paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 1 preprocessing for inscription images.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Path to the source image.")
    input_group.add_argument("--input-dir", help="Directory containing source images.")
    parser.add_argument("--output", help="Path to the processed output image.")
    parser.add_argument("--output-dir", help="Directory for batch processed output images.")
    parser.add_argument(
        "--pattern",
        default="*.jpg",
        help="Glob pattern used with --input-dir. Defaults to '*.jpg'.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s:%(name)s:%(message)s")

    if args.input:
        if not args.output:
            parser.error("--output is required when using --input")
        preprocess(args.input, args.output)
        return

    if not args.output_dir:
        parser.error("--output-dir is required when using --input-dir")
    process_directory(args.input_dir, args.output_dir, pattern=args.pattern)


if __name__ == "__main__":
    main()
