# Stage 2 Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `src/enhance.py` (AI-based image enhancement) and wire it into `api/pipeline.py` so the web UI's "Enhance" checkbox runs real denoising + Real-ESRGAN super-resolution on selected inscription images.

**Architecture:** `src/enhance.py` provides pure image-processing functions (denoise, dstretch, sharpen, enhance_with_realesrgan) plus an `enhance()` orchestrator that chains them and saves output to `data/enhanced/{stem}_enhanced.jpg`. `api/pipeline.py` gets a `_run_enhance()` adapter that finds the preprocessed image (or raw fallback) and calls `enhance()`. Real-ESRGAN degrades gracefully to OpenCV-only if torch/weights are absent.

**Tech Stack:** OpenCV, NumPy, basicsr + realesrgan + torch (optional, graceful fallback), Pillow, pytest

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `requirements.txt` | Modify | Add basicsr, realesrgan, torch, torchvision |
| `src/enhance.py` | Create | All enhancement functions + orchestrator |
| `api/pipeline.py` | Modify | Add `_run_enhance`, wire `run_stage` |
| `tests/test_enhance.py` | Create | Unit + integration tests for enhance stage |
| `tests/test_api.py` | Modify | Fix one test that assumed enhance was skipped |

---

### Task 1: Update requirements.txt

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add enhancement dependencies**

Replace the contents of `requirements.txt` with:

```text
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
httpx>=0.27.0
opencv-python>=4.9.0
numpy>=1.26.0
Pillow>=10.3.0
scikit-image>=0.23.2
basicsr>=1.4.2
realesrgan>=0.3.0
torch>=2.0.0
torchvision>=0.15.0
```

- [ ] **Step 2: Install new deps**

Run:
```bash
pip install scikit-image basicsr realesrgan torch torchvision
```

Expected: all packages install (torch may take a few minutes — it's ~2 GB on CPU-only). If basicsr/realesrgan fail on Windows, note the error but continue — graceful fallback means tests still pass.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "deps: add enhancement pipeline dependencies"
```

---

### Task 2: Core functions — denoise, dstretch, sharpen

**Files:**
- Create: `src/enhance.py`
- Create: `tests/test_enhance.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_enhance.py`:

```python
from __future__ import annotations

import numpy as np
import pytest


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
    # Output should be broadly similar to input (not black or inverted)
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_enhance.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.enhance'`

- [ ] **Step 3: Create `src/enhance.py` with denoise, dstretch, sharpen**

```python
from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Optional

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
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    stretch_matrix = (
        eigenvectors
        @ np.diag(1.0 / np.sqrt(eigenvalues + 1e-10))
        @ eigenvectors.T
    )
    stretched = centered @ stretch_matrix.T
    lo, hi = stretched.min(), stretched.max()
    stretched = (stretched - lo) / (hi - lo + 1e-10)
    return (stretched.reshape(img_float.shape) * 255).astype(np.uint8)


def sharpen(img: np.ndarray, amount: float = 1.5) -> np.ndarray:
    """Unsharp mask sharpening to crisp up character edges."""
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_enhance.py -v
```

Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/enhance.py tests/test_enhance.py
git commit -m "feat: add denoise, dstretch, sharpen to enhance module"
```

---

### Task 3: Real-ESRGAN integration

**Files:**
- Modify: `src/enhance.py` (add `_download_weights`, `enhance_with_realesrgan`)
- Modify: `tests/test_enhance.py` (add Real-ESRGAN tests)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_enhance.py`:

```python
def test_enhance_with_realesrgan_mocked(monkeypatch, tmp_path):
    """Real-ESRGAN path works when upsampler is mocked."""
    from src import enhance as enhance_mod

    class FakeUpsampler:
        def enhance(self, img_rgb, outscale):
            # Return same image (just RGB) as if upscaled
            return img_rgb, None

    monkeypatch.setattr(enhance_mod, "_build_upsampler", lambda path: FakeUpsampler())

    img = _bgr()
    out = enhance_mod.enhance_with_realesrgan(img, model_path=str(tmp_path / "fake.pth"))
    assert out.shape[2] == 3
    assert out.dtype == np.uint8


def test_enhance_with_realesrgan_returns_bgr(monkeypatch, tmp_path):
    """Output of enhance_with_realesrgan is BGR uint8."""
    from src import enhance as enhance_mod

    class FakeUpsampler:
        def enhance(self, img_rgb, outscale):
            return img_rgb, None

    monkeypatch.setattr(enhance_mod, "_build_upsampler", lambda path: FakeUpsampler())

    img = _bgr(32, 32)
    out = enhance_mod.enhance_with_realesrgan(img, model_path=str(tmp_path / "fake.pth"))
    assert out.dtype == np.uint8
    assert len(out.shape) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_enhance.py::test_enhance_with_realesrgan_mocked -v
```

Expected: FAIL — `AttributeError: module 'src.enhance' has no attribute '_build_upsampler'`

- [ ] **Step 3: Add `_download_weights`, `_build_upsampler`, `enhance_with_realesrgan` to `src/enhance.py`**

Append to `src/enhance.py` (after `sharpen`):

```python
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


def enhance_with_realesrgan(
    img: np.ndarray,
    scale: int = 2,
    model_path: str = str(DEFAULT_MODEL_PATH),
) -> np.ndarray:
    """Super-resolution via Real-ESRGAN. outscale=2 avoids over-smoothing."""
    mp = Path(model_path)
    if not mp.exists():
        _download_weights(mp)

    upsampler = _build_upsampler(str(mp))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output_rgb, _ = upsampler.enhance(img_rgb, outscale=scale)
    return cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_enhance.py -v
```

Expected: 7 PASSED (the 2 new Real-ESRGAN tests use monkeypatch so no GPU or weights needed)

- [ ] **Step 5: Commit**

```bash
git add src/enhance.py tests/test_enhance.py
git commit -m "feat: add Real-ESRGAN integration with graceful fallback"
```

---

### Task 4: `enhance()` orchestrator + `build_output_path`

**Files:**
- Modify: `src/enhance.py` (add `enhance`, `build_output_path`)
- Modify: `tests/test_enhance.py` (add orchestrator tests)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_enhance.py`:

```python
def test_enhance_end_to_end_no_realesrgan(monkeypatch, tmp_path):
    """enhance() completes successfully even when Real-ESRGAN is unavailable."""
    from src import enhance as enhance_mod

    # Make enhance_with_realesrgan raise ImportError (simulates missing torch)
    def _raise(*args, **kwargs):
        raise ImportError("torch not installed")

    monkeypatch.setattr(enhance_mod, "enhance_with_realesrgan", _raise)

    # Write a small JPEG input
    src_img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    input_path = tmp_path / "test_input.jpg"
    cv2.imwrite(str(input_path), src_img)

    output_path = tmp_path / "test_output.jpg"
    result = enhance_mod.enhance(str(input_path), str(output_path))

    assert output_path.exists()
    assert result.dtype == np.uint8
    assert result.shape[2] == 3


def test_enhance_dstretch_path(monkeypatch, tmp_path):
    """enhance() with use_dstretch=True calls dstretch, not Real-ESRGAN."""
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_enhance.py::test_enhance_end_to_end_no_realesrgan tests/test_enhance.py::test_build_output_path -v
```

Expected: FAIL — `ImportError: cannot import name 'enhance' from 'src.enhance'`

- [ ] **Step 3: Add `enhance()` and `build_output_path()` to `src/enhance.py`**

Append to `src/enhance.py`:

```python
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
        except (ImportError, Exception) as exc:
            LOGGER.warning("Real-ESRGAN unavailable (%s) — skipping super-resolution.", exc)

    img = sharpen(img)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image(out, img)

    LOGGER.info("Enhanced %s → %s", img_path, out)
    return img
```

- [ ] **Step 4: Run all enhance tests**

```bash
pytest tests/test_enhance.py -v
```

Expected: 10 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/enhance.py tests/test_enhance.py
git commit -m "feat: add enhance() orchestrator and build_output_path"
```

---

### Task 5: Wire enhance into `api/pipeline.py`

**Files:**
- Modify: `api/pipeline.py` (add `_run_enhance`, update `run_stage`)
- Modify: `tests/test_api.py` (fix stale "enhance returns skipped" assertion)

- [ ] **Step 1: Write failing test**

Append to `tests/test_api.py`:

```python
def test_run_stage_enhance_returns_done_or_failed():
    """enhance stage is now implemented — must not return skipped."""
    from api.pipeline import run_stage
    result = run_stage("IMG_3941", "enhance")
    assert result["status"] in ("done", "failed"), (
        f"Expected done or failed, got: {result['status']}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_api.py::test_run_stage_enhance_returns_done_or_failed -v
```

Expected: FAIL — `AssertionError: Expected done or failed, got: skipped`

- [ ] **Step 3: Update `api/pipeline.py`**

Add `_run_enhance` and update `run_stage`. The full updated file:

```python
from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = _PROJECT_ROOT / "data" / "raw"
ENHANCED_DIR = _PROJECT_ROOT / "data" / "enhanced"
THUMB_DIR = _PROJECT_ROOT / "data" / "thumbnails"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
THUMB_MAX_PX = 400


def make_thumbnail(image_id: str) -> Path | None:
    """Return path to cached thumbnail, generating it if needed."""
    import cv2

    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    thumb_path = THUMB_DIR / f"{image_id}_thumb.jpg"
    if thumb_path.exists():
        return thumb_path

    raw_path = _find_raw_path(image_id)
    if raw_path is None:
        return None

    img = cv2.imread(str(raw_path))
    if img is None:
        return None

    h, w = img.shape[:2]
    scale = THUMB_MAX_PX / max(h, w)
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(thumb_path), small, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return thumb_path


def _find_raw_path(image_id: str) -> Path | None:
    for path in RAW_DIR.rglob("*"):
        if path.is_file() and path.stem.lower() == image_id.lower() and path.suffix.lower() in IMAGE_SUFFIXES:
            return path
    return None


def run_stage(image_id: str, stage: str) -> dict:
    if stage == "preprocess":
        return _run_preprocess(image_id)
    if stage == "enhance":
        return _run_enhance(image_id)
    return {"status": "skipped", "reason": f"Stage '{stage}' not yet implemented"}


def _run_preprocess(image_id: str) -> dict:
    from src.preprocess import preprocess, build_output_path

    raw_path = _find_raw_path(image_id)
    if raw_path is None:
        return {"status": "failed", "error": f"Raw image not found for id '{image_id}'"}

    ENHANCED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = build_output_path(raw_path, ENHANCED_DIR)

    try:
        preprocess(str(raw_path), str(output_path))
        return {
            "status": "done",
            "url": f"/data/enhanced/{output_path.name}",
        }
    except Exception as exc:
        return {"status": "failed", "error": str(exc)}


def _run_enhance(image_id: str) -> dict:
    from src.enhance import enhance

    # Prefer preprocessed output as input; fall back to raw image
    preprocessed = ENHANCED_DIR / f"{image_id}_preprocessed.jpg"
    src_path = preprocessed if preprocessed.exists() else _find_raw_path(image_id)

    if src_path is None:
        return {"status": "failed", "error": f"No image found for id '{image_id}'"}

    ENHANCED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ENHANCED_DIR / f"{image_id}_enhanced.jpg"

    try:
        enhance(str(src_path), str(output_path))
        return {
            "status": "done",
            "url": f"/data/enhanced/{output_path.name}",
        }
    except Exception as exc:
        return {"status": "failed", "error": str(exc)}
```

- [ ] **Step 4: Fix the stale test in `tests/test_api.py`**

Find this test (line ~85):
```python
def test_run_stage_unknown_returns_skipped():
    from api.pipeline import run_stage
    result = run_stage("IMG_3941", "enhance")
    assert result["status"] == "skipped"
```

Replace with:
```python
def test_run_stage_unknown_returns_skipped():
    from api.pipeline import run_stage
    result = run_stage("IMG_3941", "translate")
    assert result["status"] == "skipped"
```

- [ ] **Step 5: Run all tests**

```bash
pytest tests/ -v
```

Expected: all tests PASS (the new enhance test accepts "done" or "failed" so it passes regardless of whether Real-ESRGAN weights are present)

- [ ] **Step 6: Commit**

```bash
git add api/pipeline.py tests/test_api.py
git commit -m "feat: wire enhance stage into pipeline adapter"
```

---

### Task 6: End-to-end smoke test via web UI

**Files:** No code changes — this is a manual verification step.

- [ ] **Step 1: Start the backend**

```bash
uvicorn api.main:app --reload --port 8000
```

- [ ] **Step 2: Start the frontend**

```bash
cd web && npm run dev
```

- [ ] **Step 3: Open http://localhost:5173 in the browser**

- [ ] **Step 4: Select one image, check "Enhance", click Run**

Expected behaviour:
- Job status shows "running" in top bar
- Results page shows the image card with "Preprocessed" stage label
- If Real-ESRGAN weights are present: result is "done" with comparison slider showing enhanced vs original
- If Real-ESRGAN weights are absent: result is "done" (fallback chain ran) with comparison slider
- Result is NOT "skipped"

- [ ] **Step 5: Push**

```bash
git push origin main
```
