"""
src/analysis.py — ECE-owned module

Performs color channel distribution calculations and histogram plots for documentation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

LOGGER = logging.getLogger("analysis")


def analyse_colour_distribution(img_path: str) -> dict:
    """Compute per-channel mean, std, skewness, and kurtosis. Returns a dict."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    # Convert to RGB (OpenCV default is BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    channels = ["Red", "Green", "Blue"]
    stats = {}

    for i, ch_name in enumerate(channels):
        ch_data = img_rgb[:, :, i].astype(np.float64)
        mean = float(np.mean(ch_data))
        std = float(np.std(ch_data))
        
        # Avoid division by zero
        std_safe = std if std > 1e-8 else 1e-10

        # Calculate Skewness and Kurtosis
        diff = ch_data - mean
        skewness = float(np.mean((diff / std_safe) ** 3))
        kurtosis = float(np.mean((diff / std_safe) ** 4) - 3)

        stats[ch_name] = {
            "mean": round(mean, 2),
            "std": round(std, 2),
            "skewness": round(skewness, 4),
            "kurtosis": round(kurtosis, 4)
        }

    return stats


def plot_histogram_comparison(
    original: np.ndarray,
    enhanced: np.ndarray,
    output_path: str,
) -> None:
    """Save side-by-side RGB histogram plot for before/after comparison."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib is not installed — skipping histogram comparison plotting")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Standardise to RGB
    orig_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB) if original.ndim == 3 else original
    enh_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB) if enhanced.ndim == 3 else enhanced

    colors = ("r", "g", "b")
    labels = ("Red", "Green", "Blue")

    # Plot original histogram
    axes[0].set_title("Original Image Histogram")
    axes[0].set_xlabel("Pixel Value")
    axes[0].set_ylabel("Frequency")
    if orig_rgb.ndim == 3:
        for i, col in enumerate(colors):
            hist = cv2.calcHist([orig_rgb], [i], None, [256], [0, 256])
            axes[0].plot(hist, color=col, label=labels[i])
        axes[0].legend()
    else:
        hist = cv2.calcHist([orig_rgb], [0], None, [256], [0, 256])
        axes[0].plot(hist, color="k", label="Grayscale")
        axes[0].legend()

    # Plot enhanced histogram
    axes[1].set_title("Enhanced Image Histogram")
    axes[1].set_xlabel("Pixel Value")
    axes[1].set_ylabel("Frequency")
    if enh_rgb.ndim == 3:
        for i, col in enumerate(colors):
            hist = cv2.calcHist([enh_rgb], [i], None, [256], [0, 256])
            axes[1].plot(hist, color=col, label=labels[i])
        axes[1].legend()
    else:
        hist = cv2.calcHist([enh_rgb], [0], None, [256], [0, 256])
        axes[1].plot(hist, color="k", label="Grayscale")
        axes[1].legend()

    plt.tight_layout()
    
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=150)
    plt.close()
    LOGGER.info("Saved histogram comparison to %s", out)
