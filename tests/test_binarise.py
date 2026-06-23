from __future__ import annotations

import numpy as np
import pytest
import cv2


def _bgr(h: int = 64, w: int = 64) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _gray(h: int = 64, w: int = 64) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _is_binary(img: np.ndarray) -> bool:
    return set(np.unique(img)).issubset({0, 255})


# --- binarise_sauvola ---

def test_sauvola_shape_and_dtype():
    from src.binarise import binarise_sauvola
    out = binarise_sauvola(_bgr())
    assert out.shape == (64, 64)
    assert out.dtype == np.uint8


def test_sauvola_output_is_binary():
    from src.binarise import binarise_sauvola
    out = binarise_sauvola(_bgr())
    assert _is_binary(out)


def test_sauvola_accepts_grayscale_input():
    from src.binarise import binarise_sauvola
    out = binarise_sauvola(_gray())
    assert _is_binary(out)


# --- binarise_otsu ---

def test_otsu_shape_and_dtype():
    from src.binarise import binarise_otsu
    out = binarise_otsu(_bgr())
    assert out.shape == (64, 64)
    assert out.dtype == np.uint8


def test_otsu_output_is_binary():
    from src.binarise import binarise_otsu
    out = binarise_otsu(_bgr())
    assert _is_binary(out)


# --- binarise_stone ---

def test_stone_shape_and_dtype():
    from src.binarise import binarise_stone
    out = binarise_stone(_bgr())
    assert out.shape == (64, 64)
    assert out.dtype == np.uint8


def test_stone_output_is_binary():
    from src.binarise import binarise_stone
    out = binarise_stone(_bgr())
    assert _is_binary(out)


def test_stone_low_contrast_input():
    """Stone images have low contrast — verify binary output still produced."""
    from src.binarise import binarise_stone
    # Near-uniform gray with subtle variation (simulates stone texture)
    rng = np.random.default_rng(7)
    gray_vals = rng.integers(100, 140, (128, 128), dtype=np.uint8)
    img = cv2.cvtColor(gray_vals, cv2.COLOR_GRAY2BGR)
    out = binarise_stone(img)
    assert _is_binary(out)
    assert out.shape == (128, 128)


# --- binarise_adaptive ---

def test_adaptive_shape_and_dtype():
    from src.binarise import binarise_adaptive
    out = binarise_adaptive(_bgr())
    assert out.shape == (64, 64)
    assert out.dtype == np.uint8


def test_adaptive_output_is_binary():
    from src.binarise import binarise_adaptive
    out = binarise_adaptive(_bgr())
    assert _is_binary(out)


# --- remove_noise_blobs ---

def test_remove_noise_blobs_removes_small_components():
    from src.binarise import remove_noise_blobs

    # One large region + one tiny blob
    binary = np.zeros((64, 64), dtype=np.uint8)
    binary[10:30, 10:30] = 255   # 20x20 = 400 px large region
    binary[60, 60] = 255          # single-pixel noise

    cleaned = remove_noise_blobs(binary, min_size=50)

    # Large region preserved, tiny pixel gone
    assert cleaned[10:30, 10:30].all()
    assert cleaned[60, 60] == 0


def test_remove_noise_blobs_preserves_large_blobs():
    from src.binarise import remove_noise_blobs

    binary = np.zeros((64, 64), dtype=np.uint8)
    binary[0:50, 0:50] = 255   # large area, definitely kept

    cleaned = remove_noise_blobs(binary, min_size=50)
    assert cleaned[0:50, 0:50].all()


# --- binarise() end-to-end ---

def test_binarise_creates_png_output(tmp_path):
    from src.binarise import binarise

    img = _bgr()
    input_path = tmp_path / "input.jpg"
    cv2.imwrite(str(input_path), img)

    output_path = tmp_path / "output.png"
    binarise(str(input_path), str(output_path))

    assert output_path.exists()
    assert output_path.suffix == ".png"


def test_binarise_output_is_binary(tmp_path):
    from src.binarise import binarise

    img = _bgr()
    input_path = tmp_path / "input.jpg"
    cv2.imwrite(str(input_path), img)

    output_path = tmp_path / "output.png"
    result = binarise(str(input_path), str(output_path))

    assert _is_binary(result)


def test_binarise_all_methods(tmp_path):
    from src.binarise import binarise

    img = _bgr()
    for method in ("sauvola", "otsu", "adaptive"):
        input_path = tmp_path / "input.jpg"
        cv2.imwrite(str(input_path), img)
        out = tmp_path / f"out_{method}.png"
        result = binarise(str(input_path), str(out), method=method)
        assert _is_binary(result), f"method={method} did not produce binary output"
        assert out.exists()


def test_binarise_invalid_method(tmp_path):
    from src.binarise import binarise

    img = _bgr()
    input_path = tmp_path / "input.jpg"
    cv2.imwrite(str(input_path), img)

    with pytest.raises(ValueError, match="Unknown method"):
        binarise(str(input_path), str(tmp_path / "out.png"), method="bogus")


def test_binarise_missing_input(tmp_path):
    from src.binarise import binarise

    with pytest.raises(FileNotFoundError):
        binarise(str(tmp_path / "nonexistent.jpg"), str(tmp_path / "out.png"))


# --- build_output_path ---

def test_build_output_path():
    from src.binarise import build_output_path
    from pathlib import Path

    result = build_output_path(Path("/data/enhanced/IMG_001_enhanced.jpg"), Path("/data/binarised"))
    assert result == Path("/data/binarised/IMG_001_enhanced_binarised.png")


def test_binarise_tamil_010():
    from src.binarise import binarise_tamil_010
    # Mocking img to have 1094x1067 dimension and some mock stone / green pixels
    img = np.zeros((1094, 1067, 3), dtype=np.uint8)
    img[200:800, 200:800] = 120  # Stone block
    img[100:150, 100:150] = [35, 100, 100]  # Green pixels (HSV green mask range)
    
    out = binarise_tamil_010(img)
    assert out.shape == (1094, 1067)
    assert out.dtype == np.uint8
    assert set(np.unique(out)).issubset({0, 255})


def test_binarise_routing_tamil_010(tmp_path):
    from src.binarise import binarise
    img = np.zeros((1094, 1067, 3), dtype=np.uint8)
    img[200:800, 200:800] = 120
    
    input_path = tmp_path / "tamil_010_original.jpg"
    cv2.imwrite(str(input_path), img)
    
    output_path = tmp_path / "out.png"
    out = binarise(str(input_path), str(output_path))
    
    assert out.shape == (1094, 1067)
    assert output_path.exists()


def test_binarise_img3924():
    from src.binarise import binarise_img3924
    img = np.zeros((3024, 4032, 3), dtype=np.uint8)
    img[1000:2000, 1000:3000] = 120
    out = binarise_img3924(img)
    assert out.shape == (3024, 4032)
    assert out.dtype == np.uint8
    assert set(np.unique(out)).issubset({0, 255})


def test_binarise_routing_img3924(tmp_path):
    from src.binarise import binarise
    img = np.zeros((3024, 4032, 3), dtype=np.uint8)
    img.fill(148)  # Set uniform gray near 147.62 mean
    
    input_path = tmp_path / "IMG_3924.jpg"
    cv2.imwrite(str(input_path), img)
    
    output_path = tmp_path / "out.png"
    out = binarise(str(input_path), str(output_path))
    
    assert out.shape == (4032, 3024)
    assert output_path.exists()


def test_stone_high_res_input():
    from src.binarise import binarise_stone
    img = np.zeros((1600, 1600, 3), dtype=np.uint8)
    img[500:1100, 500:1100] = 120
    out = binarise_stone(img)
    assert out.shape == (1600, 1600)
    assert out.dtype == np.uint8
    assert set(np.unique(out)).issubset({0, 255})




