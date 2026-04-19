# Stage 2 — Enhancement Implementation Design

## Goal

Implement `src/enhance.py` (Stage 2 of the IDP pipeline) and wire it into `api/pipeline.py` so the web UI's "Enhance" checkbox runs real AI-based image enhancement on selected inscription images.

## Architecture

```
Raw/preprocessed image
        ↓
  denoise()           ← cv2.fastNlMeansDenoisingColored
        ↓
  enhance_with_realesrgan()   ← Real-ESRGAN outscale=2
  OR dstretch()               ← if use_dstretch=True (cave/rock art)
        ↓
  sharpen()           ← unsharp mask
        ↓
  data/enhanced/{stem}_enhanced.jpg
```

Graceful fallback: if `torch` is not installed or model weights are absent, the chain runs `denoise → sharpen` via OpenCV only and logs a warning. Never raises.

## Input Chaining

`_run_enhance(image_id)` in `api/pipeline.py` checks for the preprocessed output first:
1. Look for `data/enhanced/{stem}_preprocessed.jpg` (output of Stage 1)
2. Fall back to raw image if preprocessing hasn't been run

## Files Changed

| File | Action |
|---|---|
| `src/enhance.py` | Create — full enhancement chain |
| `api/pipeline.py` | Modify — add `_run_enhance`, wire `run_stage` |
| `requirements.txt` | Modify — add `basicsr`, `realesrgan`, `torch>=2.0.0`, `torchvision` |

## `src/enhance.py` Public API

```python
def denoise(img: np.ndarray, strength: int = 10) -> np.ndarray:
    """cv2.fastNlMeansDenoisingColored. strength=10 mild, 20 heavy."""

def enhance_with_realesrgan(
    img: np.ndarray,
    scale: int = 2,
    model_path: str = "models/weights/RealESRGAN_x4plus.pth"
) -> np.ndarray:
    """Super-resolution via Real-ESRGAN. outscale=2 to avoid over-smoothing."""

def dstretch(img: np.ndarray, colour_space: str = "LAB") -> np.ndarray:
    """Decorrelation stretch for cave/rock paintings. NumPy only."""

def sharpen(img: np.ndarray, amount: float = 1.5) -> np.ndarray:
    """Unsharp mask sharpening."""

def enhance(
    img_path: str,
    output_path: str,
    use_dstretch: bool = False
) -> np.ndarray:
    """Full chain. Gracefully degrades if Real-ESRGAN unavailable."""

def build_output_path(input_path: Path, output_dir: Path) -> Path:
    """Returns output_dir / {stem}_enhanced.jpg"""
```

## `api/pipeline.py` Changes

Add `BINARISED_DIR` constant and `_run_enhance(image_id) -> dict`:
- Finds preprocessed image or falls back to raw
- Calls `src.enhance.enhance(str(src_path), str(output_path))`
- Returns `{"status": "done", "url": "/data/enhanced/{name}"}` on success
- Returns `{"status": "failed", "error": str(exc)}` on failure

Update `run_stage` to route `"enhance"` → `_run_enhance`.

## Model Weights

`enhance_with_realesrgan` auto-downloads `RealESRGAN_x4plus.pth` to `models/weights/` on first call if the file is absent, using `urllib.request`. The weights URL is the official GitHub release asset.

## Real-ESRGAN Config (per AGENTS.md)

```python
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
    scale=4, model_path=model_path, model=model,
    tile=400, tile_pad=10, pre_pad=0, half=False
)
# enhance at outscale=2 (4x model, 2x output)
output, _ = upsampler.enhance(img_rgb, outscale=2)
```

## Output

- File: `data/enhanced/{stem}_enhanced.jpg` at JPEG quality 95
- Web URL served via existing `/data` StaticFiles mount
- Stage result returned to job store: `{"status": "done", "url": "/data/enhanced/..."}`

## Error Handling

- Input file not found → `{"status": "failed", "error": "..."}`
- Real-ESRGAN unavailable → fallback chain, result still `"done"`
- Any other exception → `{"status": "failed", "error": str(exc)}`

## Testing

- `tests/test_enhance.py` with a synthetic 64×64 BGR test image
- Test `denoise`, `dstretch`, `sharpen` output shapes and dtypes
- Test `enhance()` end-to-end with a small synthetic image (Real-ESRGAN skipped via monkeypatch)
- Test `_run_enhance` API adapter returns correct dict structure
