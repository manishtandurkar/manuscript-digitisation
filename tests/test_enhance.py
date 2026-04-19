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
