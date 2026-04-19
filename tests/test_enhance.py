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


def test_enhance_with_realesrgan_mocked(monkeypatch, tmp_path):
    """Real-ESRGAN path works when upsampler is mocked."""
    from src import enhance as enhance_mod

    class FakeUpsampler:
        def enhance(self, img_rgb, outscale):
            return img_rgb, None

    monkeypatch.setattr(enhance_mod, "_build_upsampler", lambda path: FakeUpsampler())

    fake_pth = tmp_path / "fake.pth"
    fake_pth.write_bytes(b"fake")  # create the file so download is skipped
    img = _bgr()
    out = enhance_mod.enhance_with_realesrgan(img, model_path=str(fake_pth))
    assert out.shape[2] == 3
    assert out.dtype == np.uint8


def test_enhance_with_realesrgan_returns_bgr(monkeypatch, tmp_path):
    """Output is BGR uint8."""
    from src import enhance as enhance_mod

    class FakeUpsampler:
        def enhance(self, img_rgb, outscale):
            return img_rgb, None

    monkeypatch.setattr(enhance_mod, "_build_upsampler", lambda path: FakeUpsampler())

    fake_pth = tmp_path / "fake.pth"
    fake_pth.write_bytes(b"fake")  # create the file so download is skipped
    img = _bgr(32, 32)
    out = enhance_mod.enhance_with_realesrgan(img, model_path=str(fake_pth))
    assert out.dtype == np.uint8
    assert len(out.shape) == 3


def test_enhance_end_to_end_no_realesrgan(monkeypatch, tmp_path):
    """enhance() completes successfully even when Real-ESRGAN is unavailable."""
    from src import enhance as enhance_mod

    def _raise(*args, **kwargs):
        raise ImportError("torch not installed")

    monkeypatch.setattr(enhance_mod, "enhance_with_realesrgan", _raise)

    src_img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    input_path = tmp_path / "test_input.jpg"
    cv2.imwrite(str(input_path), src_img)

    output_path = tmp_path / "test_output.jpg"
    result = enhance_mod.enhance(str(input_path), str(output_path))

    assert output_path.exists()
    assert result.dtype == np.uint8
    assert result.shape[2] == 3


def test_enhance_dstretch_path(monkeypatch, tmp_path):
    """enhance() with use_dstretch=True calls dstretch."""
    from src import enhance as enhance_mod

    called = {}
    real_dstretch = enhance_mod.dstretch

    def tracking_dstretch(img, colour_space="LAB"):
        called["dstretch"] = True
        return real_dstretch(img, colour_space)

    monkeypatch.setattr(enhance_mod, "dstretch", tracking_dstretch)

    src_img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    input_path = tmp_path / "cave.jpg"
    cv2.imwrite(str(input_path), src_img)

    output_path = tmp_path / "cave_enhanced.jpg"
    enhance_mod.enhance(str(input_path), str(output_path), use_dstretch=True)

    assert called.get("dstretch") is True
    assert output_path.exists()


def test_build_output_path():
    from src.enhance import build_output_path
    from pathlib import Path
    result = build_output_path(Path("/data/raw/IMG_001.jpg"), Path("/data/enhanced"))
    assert result == Path("/data/enhanced/IMG_001_enhanced.jpg")
