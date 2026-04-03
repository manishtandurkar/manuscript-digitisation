from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import read_image_metadata, save_image

LOGGER = logging.getLogger("preprocess")


def load_image(path: str) -> np.ndarray:
    """Load image preserving colour space. Return BGR numpy array."""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {path}")
    return image


def normalise_brightness(img: np.ndarray) -> np.ndarray:
    """CLAHE histogram equalisation for uneven lighting."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def auto_white_balance(img: np.ndarray) -> np.ndarray:
    """Grey-world assumption white balance correction."""
    img_float = img.astype(np.float32)
    channel_means = img_float.reshape(-1, 3).mean(axis=0)
    overall_mean = float(channel_means.mean())
    scale = overall_mean / np.maximum(channel_means, 1e-6)
    balanced = img_float * scale.reshape(1, 1, 3)
    return np.clip(balanced, 0, 255).astype(np.uint8)


def _estimate_rotation_angle(img: np.ndarray, min_lines: int = 10) -> Tuple[float, int]:
    """
    Estimate rotation angle using Hough line transform.
    
    Args:
        img: Input image
        min_lines: Minimum number of valid lines required for reliable angle estimation
        
    Returns:
        (angle in degrees, number of lines used)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    min_line_length = max(int(min(img.shape[:2]) * 0.25), 100)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=min_line_length,
        maxLineGap=20,
    )

    if lines is None:
        return 0.0, 0

    angles = []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = line
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        while angle <= -90:
            angle += 180
        while angle > 90:
            angle -= 180

        # Only consider near-horizontal or near-vertical lines
        if abs(angle) <= 15:  # More conservative: ±15 degrees instead of ±20
            angles.append(angle)

    if len(angles) < min_lines:
        # Not enough confidence - don't rotate
        return 0.0, len(angles)

    return float(np.median(angles)), len(angles)


def _deskew_with_metadata(img: np.ndarray, min_lines: int = 10, max_angle: float = 10.0) -> Tuple[np.ndarray, float, int]:
    """
    Deskew image with metadata tracking.
    
    Args:
        img: Input image
        min_lines: Minimum number of lines required for reliable rotation detection
        max_angle: Maximum rotation angle to apply (safety limit)
        
    Returns:
        (deskewed image, angle applied, number of lines used)
    """
    angle, line_count = _estimate_rotation_angle(img, min_lines=min_lines)
    
    # Safety checks before rotating
    if abs(angle) < 0.3:
        # Angle too small to matter
        return img.copy(), 0.0, line_count
    
    if abs(angle) > max_angle:
        # Angle suspiciously large - likely incorrect detection
        LOGGER.warning(
            "Detected angle %.2f° exceeds max_angle=%.1f° (based on %d lines). Skipping rotation.",
            angle, max_angle, line_count
        )
        return img.copy(), 0.0, line_count

    height, width = img.shape[:2]
    centre = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
    rotated = cv2.warpAffine(
        img,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, angle, line_count


def deskew(img: np.ndarray, enable: bool = True, min_lines: int = 10, max_angle: float = 10.0) -> np.ndarray:
    """
    Detect and correct rotation angle using Hough line transform.
    
    Args:
        img: Input image
        enable: Whether to enable deskewing (set False to skip)
        min_lines: Minimum number of lines required for reliable rotation
        max_angle: Maximum rotation angle to apply (degrees)
        
    Returns:
        Deskewed image (or original if enable=False or detection unreliable)
    """
    if not enable:
        return img.copy()
    
    corrected, _, _ = _deskew_with_metadata(img, min_lines=min_lines, max_angle=max_angle)
    return corrected


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


def preprocess(
    img_path: str,
    output_path: str,
    enable_deskew: bool = True,
    deskew_min_lines: int = 10,
    deskew_max_angle: float = 10.0,
    save_access_copy: bool = True,
) -> np.ndarray:
    """
    Run full preprocessing chain and save result.
    
    Args:
        img_path: Path to input image
        output_path: Path to output master image (TIFF recommended)
        enable_deskew: Whether to enable automatic deskewing
        deskew_min_lines: Minimum lines required for reliable deskew
        deskew_max_angle: Maximum rotation angle to apply (safety limit)
        save_access_copy: Whether to save a JPEG access copy alongside master
        
    Returns:
        Preprocessed image as numpy array
    """
    input_path = Path(img_path)
    metadata = read_image_metadata(input_path)
    original = load_image(str(input_path))
    before_height, before_width = original.shape[:2]

    processed = normalise_brightness(original)
    processed = auto_white_balance(processed)
    processed, angle, line_count = _deskew_with_metadata(
        processed, 
        min_lines=deskew_min_lines, 
        max_angle=deskew_max_angle
    ) if enable_deskew else (processed, 0.0, 0)
    processed, crop_box = _crop_borders_with_metadata(processed)

    # Save master output
    master_output_path = Path(output_path)
    save_image(master_output_path, processed, metadata=metadata)
    
    # Optionally save access copy
    access_output_path = None
    if save_access_copy:
        access_output_path = build_access_output_path(master_output_path)
        save_image(access_output_path, processed, metadata=metadata)

    after_height, after_width = processed.shape[:2]
    LOGGER.info(
        "Preprocessed %s | before=%sx%s after=%sx%s angle=%.2f° (lines=%s) crop=%s deskew=%s master=%s access=%s",
        input_path.name,
        before_width,
        before_height,
        after_width,
        after_height,
        angle,
        line_count,
        crop_box,
        "enabled" if enable_deskew else "disabled",
        master_output_path,
        access_output_path or "not saved",
    )
    return processed


def build_output_path(input_path: str | Path, output_dir: str | Path) -> Path:
    """Build the output TIFF path for a preprocessed image."""
    source = Path(input_path)
    target_dir = Path(output_dir)
    return target_dir / f"{source.stem}_preprocessed.tif"


def build_access_output_path(master_output_path: str | Path) -> Path:
    """Build the JPEG access-copy path for a master preprocessing output."""
    master_path = Path(master_output_path)
    return master_path.with_suffix(".jpg")


def process_directory(
    input_dir: str,
    output_dir: str,
    pattern: str = "*.jpg",
    enable_deskew: bool = True,
    save_access_copy: bool = True,
) -> list[Path]:
    """
    Preprocess every matching image in a directory.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory for output images
        pattern: Glob pattern to match (default: '*.jpg')
        enable_deskew: Whether to enable automatic deskewing
        save_access_copy: Whether to save JPEG access copies
        
    Returns:
        List of output master file paths
    """
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
        preprocess(
            str(input_path), 
            str(output_path),
            enable_deskew=enable_deskew,
            save_access_copy=save_access_copy,
        )
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
        "--no-deskew",
        action="store_true",
        help="Disable automatic rotation correction (use if images are getting incorrectly tilted).",
    )
    parser.add_argument(
        "--deskew-min-lines",
        type=int,
        default=10,
        help="Minimum lines required for reliable deskew detection (default: 10).",
    )
    parser.add_argument(
        "--deskew-max-angle",
        type=float,
        default=10.0,
        help="Maximum rotation angle to apply in degrees (default: 10.0).",
    )
    parser.add_argument(
        "--no-access-copy",
        action="store_true",
        help="Don't save JPEG access copies (save only master TIFF files for efficiency).",
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

    enable_deskew = not args.no_deskew
    save_access_copy = not args.no_access_copy

    if args.input:
        if not args.output:
            parser.error("--output is required when using --input")
        preprocess(
            args.input, 
            args.output,
            enable_deskew=enable_deskew,
            deskew_min_lines=args.deskew_min_lines,
            deskew_max_angle=args.deskew_max_angle,
            save_access_copy=save_access_copy,
        )
        return

    if not args.output_dir:
        parser.error("--output-dir is required when using --input-dir")
    process_directory(
        args.input_dir, 
        args.output_dir, 
        pattern=args.pattern,
        enable_deskew=enable_deskew,
        save_access_copy=save_access_copy,
    )


if __name__ == "__main__":
    main()
