from __future__ import annotations

import numpy as np
import pytest
import cv2


def _bgr(h: int = 64, w: int = 64) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_denoise_preserves_shape_and_dtype():
    from src.enhance import denoise
    img = _bgr()
    out = denoise(img, strength=5)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_denoise_mild_does_not_flip_image():
    from src.enhance import denoise
    img = _bgr()
    out = denoise(img, strength=5)
    assert int(out.mean()) > 10


def test_dstretch_preserves_shape_and_dtype():
    from src.enhance import dstretch
    img = _bgr()
    out = dstretch(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_dstretch_output_is_full_range():
    from src.enhance import dstretch
    img = _bgr()
    out = dstretch(img)
    assert out.min() >= 0
    assert out.max() <= 255


def test_sharpen_preserves_shape_and_dtype():
    from src.enhance import sharpen
    img = _bgr()
    out = sharpen(img, amount=1.5)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_dstretch_solid_colour_does_not_crash():
    from src.enhance import dstretch
    solid = np.full((64, 64, 3), 128, dtype=np.uint8)
    out = dstretch(solid)
    assert out.shape == solid.shape
    assert out.dtype == np.uint8


def test_dstretch_increases_contrast():
    from src.enhance import dstretch
    # Low-contrast image: values clustered around 128 with tiny variation
    rng = np.random.default_rng(7)
    low_contrast = (128 + rng.integers(-5, 6, (64, 64, 3))).astype(np.uint8)
    out = dstretch(low_contrast)
    assert float(out.std()) > float(low_contrast.std()), (
        "dstretch should increase contrast on a low-contrast image"
    )


def test_sharpen_increases_sharpness():
    from src.enhance import sharpen
    import cv2 as cv2_local
    # Create a blurry image
    rng = np.random.default_rng(11)
    sharp_src = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    blurry = cv2_local.GaussianBlur(sharp_src, (0, 0), sigmaX=5)
    sharpened = sharpen(blurry, amount=1.5)
    # Laplacian variance measures sharpness — sharpened should be higher than blurry
    def lap_var(img):
        gray = cv2_local.cvtColor(img, cv2_local.COLOR_BGR2GRAY)
        return float(cv2_local.Laplacian(gray, cv2_local.CV_64F).var())
    assert lap_var(sharpened) > lap_var(blurry)
