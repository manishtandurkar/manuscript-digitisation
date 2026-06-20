"""
src/filters.py — ECE-owned module

Custom spatial and frequency-domain filters for script and inscription processing.
"""

from __future__ import annotations

import cv2
import numpy as np


def gabor_filter_bank(
    img: np.ndarray,
    frequencies: list[float] = [0.1, 0.2, 0.4],
    orientations: int = 8,
) -> np.ndarray:
    """Apply Gabor filter bank to separate text texture from background."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    accum = np.zeros_like(gray, dtype=np.float32)

    for freq in frequencies:
        lambd = 1.0 / max(freq, 1e-5)
        # Generate kernels for different orientations
        theta_vals = np.linspace(0, np.pi, orientations, endpoint=False)
        for theta in theta_vals:
            # Sigma=5, lambd=wavelength, gamma=aspect, psi=phase offset
            kern = cv2.getGaborKernel(
                ksize=(21, 21),
                sigma=5.0,
                theta=theta,
                lambd=lambd,
                gamma=0.5,
                psi=0,
                ktype=cv2.CV_32F
            )
            filtered = cv2.filter2D(gray, cv2.CV_32F, kern)
            np.maximum(accum, filtered, out=accum)

    return np.clip(accum, 0, 255).astype(np.uint8)


def directional_edge_enhance(img: np.ndarray, angle_deg: float = 45.0) -> np.ndarray:
    """Enhance edges in a specific direction — useful for carved inscriptions."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    theta = np.deg2rad(angle_deg)

    # Compute Sobel gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Project gradient vectors in the target direction
    proj = sobelx * np.cos(theta) + sobely * np.sin(theta)
    proj_abs = np.abs(proj)

    # Normalise output to 0-255 range
    norm_img = cv2.normalize(proj_abs, None, 0, 255, cv2.NORM_MINMAX)
    return norm_img.astype(np.uint8)


def remove_periodic_noise_fft(img: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """Use FFT to detect and remove periodic noise (scanner line artifacts)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

    # Perform 2D Fast Fourier Transform
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-10)

    # Find high-amplitude spikes away from the center DC frequency
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    mag_max = magnitude_spectrum.max()

    # Zero out noise spikes
    for y in range(h):
        for x in range(w):
            # Exclude low frequency central components
            if abs(y - cy) < 12 and abs(x - cx) < 12:
                continue
            # If the magnitude exceeds the threshold percentile, mask it
            if magnitude_spectrum[y, x] > mag_max * (1.0 - threshold):
                fshift[y, x] = 0

    # Perform Inverse FFT
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return np.clip(img_back, 0, 255).astype(np.uint8)
