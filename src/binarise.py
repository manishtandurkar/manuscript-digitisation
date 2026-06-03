from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

LOGGER = logging.getLogger("binarise")

_MODELS_DIR = Path(__file__).parent.parent / "models" / "weights"
_CONFIDENCE_THRESHOLD = 0.65
_MODEL_CACHE: dict[str, Any] = {}

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ─── DL model definitions (only when torch present) ───────────────────────────

if _TORCH_AVAILABLE:

    class _DoubleConv(nn.Module):
        def __init__(self, in_ch: int, out_ch: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)

    class _LightUNet(nn.Module):
        """
        Lightweight U-Net for document binarisation.
        Train on THPLMD grayscale->binary pairs (input: 1xHxW float, output: 1xHxW sigmoid).
        Expected weights file: models/weights/unet_binarise.pth
        """
        _CH = [1, 32, 64, 128, 256]

        def __init__(self) -> None:
            super().__init__()
            c = self._CH
            self.enc1 = _DoubleConv(c[0], c[1])
            self.enc2 = _DoubleConv(c[1], c[2])
            self.enc3 = _DoubleConv(c[2], c[3])
            self.bottleneck = _DoubleConv(c[3], c[4])
            self.up3 = nn.ConvTranspose2d(c[4], c[3], 2, stride=2)
            self.dec3 = _DoubleConv(c[4], c[3])
            self.up2 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
            self.dec2 = _DoubleConv(c[3], c[2])
            self.up1 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
            self.dec1 = _DoubleConv(c[2], c[1])
            self.head = nn.Conv2d(c[1], 1, 1)
            self.pool = nn.MaxPool2d(2)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            b = self.bottleneck(self.pool(e3))
            d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return torch.sigmoid(self.head(d1))

    class _PatchEmbed(nn.Module):
        def __init__(self, patch_size: int = 8, embed_dim: int = 256) -> None:
            super().__init__()
            self.patch_size = patch_size
            self.proj = nn.Linear(patch_size * patch_size, embed_dim)

        def forward(
            self, x: "torch.Tensor"
        ) -> "tuple[torch.Tensor, int, int]":
            ps = self.patch_size
            x = x.unfold(2, ps, ps).unfold(3, ps, ps)
            hp, wp = x.shape[2], x.shape[3]
            B = x.shape[0]
            x = x.contiguous().view(B, hp * wp, ps * ps)
            return self.proj(x), hp, wp

    class _DocEnTr(nn.Module):
        """
        Simplified DocEnTr: patch-ViT encoder + CNN decoder for binarisation.
        Ref: El-Hajj & Barakat, ArXiv 2209.09921.
        Train on THPLMD grayscale->binary pairs (input: 1xHxW float, output: 1xHxW sigmoid).
        H and W must be multiples of patch_size (padding handled in _dl_infer).
        Expected weights file: models/weights/docentr_binarise.pth
        """

        def __init__(
            self,
            patch_size: int = 8,
            embed_dim: int = 256,
            num_layers: int = 4,
            num_heads: int = 8,
        ) -> None:
            super().__init__()
            self.patch_size = patch_size
            self.patch_embed = _PatchEmbed(patch_size, embed_dim)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=512,
                dropout=0.0,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
            self.dec_proj = nn.Linear(embed_dim, patch_size * patch_size)
            self.refine = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 1),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            B = x.shape[0]
            patches, hp, wp = self.patch_embed(x)
            tokens = self.transformer(patches)
            ps = self.patch_size
            decoded = self.dec_proj(tokens).view(B, hp, wp, ps, ps)
            decoded = decoded.permute(0, 1, 3, 2, 4).contiguous()
            decoded = decoded.view(B, 1, hp * ps, wp * ps)
            return torch.sigmoid(self.refine(decoded))


# ─── DL inference helpers ─────────────────────────────────────────────────────

def _pad_to_multiple(
    arr: np.ndarray, multiple: int
) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = arr.shape[:2]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    return np.pad(arr, ((0, ph), (0, pw)), mode="reflect"), (ph, pw)


def _binary_entropy_confidence(prob: np.ndarray) -> float:
    """Mean certainty over all pixels: 1 = fully certain, 0 = fully uncertain."""
    eps = 1e-7
    p = np.clip(prob, eps, 1 - eps)
    entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return float(1.0 - entropy.mean())


def _load_dl_model(name: str, weights_path: Path) -> "nn.Module | None":
    if name in _MODEL_CACHE:
        return _MODEL_CACHE[name]

    if not _TORCH_AVAILABLE:
        LOGGER.warning("torch not installed — DL binarisation unavailable")
        return None

    if not weights_path.exists():
        LOGGER.warning("No weights at %s — falling back to Sauvola", weights_path)
        return None

    model: nn.Module = _LightUNet() if name == "unet" else _DocEnTr()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(str(weights_path), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    _MODEL_CACHE[name] = model
    LOGGER.info("Loaded %s from %s (device=%s)", name, weights_path, device)
    return model


def _dl_infer(
    img: np.ndarray, model_name: str, weights_path: Path
) -> tuple[np.ndarray | None, float]:
    """Returns (prob_map HxW float32, confidence). prob_map None if model unavailable."""
    model = _load_dl_model(model_name, weights_path)
    if model is None:
        return None, 0.0

    gray = _to_gray(img).astype(np.float32) / 255.0
    padded, _ = _pad_to_multiple(gray, 8)
    tensor = torch.from_numpy(padded[None, None]).float()
    device = next(model.parameters()).device
    tensor = tensor.to(device)

    with torch.no_grad():
        prob = model(tensor)

    prob_np: np.ndarray = prob.squeeze().cpu().numpy()
    h, w = gray.shape
    prob_np = prob_np[:h, :w]
    return prob_np, _binary_entropy_confidence(prob_np)


def _prob_to_binary(prob: np.ndarray) -> np.ndarray:
    binary = (prob > 0.5).astype(np.uint8) * 255
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


# ─── classical methods ────────────────────────────────────────────────────────

def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _clahe(gray: np.ndarray) -> np.ndarray:
    """CLAHE equalization — normalises uneven illumination before thresholding."""
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


# ─── NEW: Adaptive parameter selection ────────────────────────────────────────

def _image_stats(gray: np.ndarray) -> dict:
    """Compute brightness/contrast stats used for adaptive param selection."""
    mean = float(gray.mean())
    std  = float(gray.std())
    h, w = gray.shape
    return {"mean": mean, "std": std, "h": h, "w": w, "shorter": min(h, w)}


def _adaptive_sauvola_params(gray: np.ndarray) -> tuple[int, float]:
    """
    Dynamically tune Sauvola (window_size, k) from image statistics.

    Rules (empirically derived from inscription image diagnostics):
      - High-res images (shorter side > 1000px): larger window captures more context
      - Low-contrast images (std < 30): increase k to be more aggressive
      - Very dark images (mean < 60): decrease k slightly to avoid over-thresholding
      - Palm-leaf fibre noise: small window (31–41) with moderate k (0.15–0.20)

    Returns (window_size, k) — window_size is always odd.
    """
    s = _image_stats(gray)
    shorter = s["shorter"]
    mean    = s["mean"]
    std     = s["std"]

    # Base window: ~1/20 of shorter dimension, clamped [15, 71]
    ws = max(15, min(71, shorter // 20))
    if ws % 2 == 0:
        ws += 1

    # Scale window up for high-res images
    if shorter > 1500:
        ws = min(81, ws + 10)
    elif shorter > 800:
        ws = min(61, ws + 6)

    # Base k
    k = 0.20

    # Low contrast → be more aggressive (raise k)
    if std < 25:
        k = 0.30
    elif std < 40:
        k = 0.25

    # Very dark image (rubbing/estampage style) → moderate k
    if mean < 60:
        k = max(k, 0.18)

    # High brightness (pale stone, outdoor) → lower k to catch faint strokes
    if mean > 160:
        k = min(k, 0.15)

    return ws, k


def binarise_sauvola(
    img: np.ndarray,
    window_size: int | None = None,
    k: float | None = None,
) -> np.ndarray:
    """Sauvola local thresholding with adaptive parameter selection.
    Output: white text, black background.
    """
    from skimage.filters import threshold_sauvola

    gray = _clahe(_to_gray(img))
    ws, k_auto = _adaptive_sauvola_params(gray)

    ws_final = window_size if window_size is not None else ws
    k_final  = k           if k          is not None else k_auto

    thresh  = threshold_sauvola(gray, window_size=ws_final, k=k_final)
    binary  = (gray < thresh).astype(np.uint8) * 255

    # Morphological cleanup: close small gaps in strokes
    stroke_kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, stroke_kernel)
    return binary


def binarise_otsu(img: np.ndarray) -> np.ndarray:
    """Otsu global thresholding. Output: white text, black background."""
    gray = _clahe(_to_gray(img))
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def binarise_adaptive(img: np.ndarray) -> np.ndarray:
    """OpenCV adaptive mean thresholding. Output: white text, black background."""
    gray = _clahe(_to_gray(img))
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 8
    )
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def binarise_stone(img: np.ndarray) -> np.ndarray:
    """
    Stone inscription binarisation — adaptive dual-path.

    Path A (black-hat): deep carvings with strong shadow contrast.
    Path B (CLAHE + adaptive Sauvola): shallow/faint/weathered carvings.
    Selection: whichever path produces more large connected components.

    Morphological kernels are scaled to image resolution so strokes
    are closed without merging adjacent characters.
    """
    gray = _to_gray(img)
    h, w = gray.shape
    shorter = min(h, w)

    # Morphological kernel sizes scaled to resolution
    close_k = max(3, shorter // 300)   # 3–7 px typically
    open_k  = max(2, shorter // 500)   # 2–4 px typically

    # --- Path A: black-hat (deep carvings) ---
    smooth_a = cv2.GaussianBlur(gray, (0, 0), sigmaX=2, sigmaY=2)
    bh_k = max(21, shorter // 12)
    if bh_k % 2 == 0:
        bh_k += 1
    kernel_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bh_k, bh_k))
    black_hat = cv2.morphologyEx(smooth_a, cv2.MORPH_BLACKHAT, kernel_bh)
    black_hat = cv2.normalize(black_hat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Adaptive threshold percentile based on std
    stats = _image_stats(gray)
    pct = 65 if stats["std"] > 35 else 75
    thresh_val = max(int(np.percentile(black_hat, pct)), 15)
    _, binary_a = cv2.threshold(black_hat, thresh_val, 255, cv2.THRESH_BINARY)
    binary_a = cv2.morphologyEx(binary_a, cv2.MORPH_OPEN,  np.ones((open_k, open_k), np.uint8))
    binary_a = cv2.morphologyEx(binary_a, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))

    # --- Path B: CLAHE + adaptive Sauvola (shallow/weathered) ---
    from skimage.filters import threshold_sauvola
    smooth_b = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.5, sigmaY=1.5)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq = clahe.apply(smooth_b)
    ws, k = _adaptive_sauvola_params(eq)
    thresh_s = threshold_sauvola(eq, window_size=ws, k=k)
    binary_b = (eq < thresh_s).astype(np.uint8) * 255
    binary_b = cv2.morphologyEx(binary_b, cv2.MORPH_OPEN,  np.ones((open_k, open_k), np.uint8))
    binary_b = cv2.morphologyEx(binary_b, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))

    # --- Path C: Otsu fallback for high-contrast rubbings ---
    # Rubbings (image1.jpeg style) are already near-binary; Otsu handles them cleanly
    _, binary_c = cv2.threshold(
        cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray),
        0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    binary_c = cv2.morphologyEx(binary_c, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))

    def _count_large(bin_img: np.ndarray) -> int:
        min_area = max(50, (shorter // 100) ** 2)
        n, _, stats_cc, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
        return sum(1 for i in range(1, n) if stats_cc[i, cv2.CC_STAT_AREA] >= min_area)

    scores = [_count_large(b) for b in (binary_a, binary_b, binary_c)]
    best   = [binary_a, binary_b, binary_c][scores.index(max(scores))]
    LOGGER.debug("stone paths scores A=%d B=%d C=%d", *scores)
    return best


def binarise_palm_leaf(img: np.ndarray) -> np.ndarray:
    """
    Palm-leaf manuscript binarisation.

    Uses adaptive Gaussian thresholding on raw gray (no CLAHE — destroys fibre contrast).
    Block size and C are derived from image resolution and contrast stats.
    Output: white text, black background.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    shorter = min(h, w)

    # Adaptive block size: ~1/15 of shorter side, must be odd, clamped [21, 51]
    block = max(21, min(51, (shorter // 15) | 1))
    if block % 2 == 0:
        block += 1

    # C offset: higher for low-contrast leaves (faded ink)
    stats = _image_stats(gray)
    C = 8 if stats["std"] > 30 else 12

    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block, C
    )

    # Flood-fill corners to remove edge border noise
    mask = np.zeros((h + 2, w + 2), np.uint8)
    for corner in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
        cv2.floodFill(binary, mask, (corner[1], corner[0]), 0)

    # Fine close (2×2) preserves thin ink strokes on fibre background
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

    # Open slightly to disconnect fibre noise from strokes
    open_k = max(1, shorter // 800)
    if open_k > 1:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))

    return binary


# ─── DL methods (public) ─────────────────────────────────────────────────────

def binarise_unet(
    img: np.ndarray,
    weights_path: Path | None = None,
) -> np.ndarray:
    """
    Lightweight U-Net binarisation.
    Falls back to Sauvola when confidence < _CONFIDENCE_THRESHOLD or weights absent.
    """
    wp = Path(weights_path) if weights_path else _MODELS_DIR / "unet_binarise.pth"
    prob, conf = _dl_infer(img, "unet", wp)
    if prob is None or conf < _CONFIDENCE_THRESHOLD:
        LOGGER.info("unet confidence %.3f below threshold — Sauvola fallback", conf)
        return binarise_sauvola(img)
    return _prob_to_binary(prob)


def binarise_docentr(
    img: np.ndarray,
    weights_path: Path | None = None,
) -> np.ndarray:
    """
    DocEnTr (patch-ViT) binarisation.
    Falls back to Sauvola when confidence < _CONFIDENCE_THRESHOLD or weights absent.
    """
    wp = Path(weights_path) if weights_path else _MODELS_DIR / "docentr_binarise.pth"
    prob, conf = _dl_infer(img, "docentr", wp)
    if prob is None or conf < _CONFIDENCE_THRESHOLD:
        LOGGER.info("docentr confidence %.3f below threshold — Sauvola fallback", conf)
        return binarise_sauvola(img)
    return _prob_to_binary(prob)


# ─── noise removal ────────────────────────────────────────────────────────────

def remove_noise_blobs(
    binary: np.ndarray,
    min_size: int | None = None,
    min_length: int | None = None,
) -> np.ndarray:
    """Remove small disconnected components. Parameters auto-scaled if not given."""
    h, w = binary.shape
    shorter = min(h, w)

    # Auto-scale: noise blob threshold ~= (shorter/200)^2, min 20
    if min_size is None:
        min_size = max(20, (shorter // 200) ** 2)
    if min_length is None:
        min_length = max(10, shorter // 150)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    cleaned = np.zeros_like(binary)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        cw   = int(stats[label, cv2.CC_STAT_WIDTH])
        ch   = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area >= min_size or max(cw, ch) >= min_length:
            cleaned[labels == label] = 255
    return cleaned


# ---------------------------------------------------------------------------
# Document-type detection
# Palm leaf manuscripts: warm orange/tan background, high saturation, hue 8-30.
# Stone inscriptions: near-achromatic, low saturation.
# ---------------------------------------------------------------------------

def detect_document_type(img: np.ndarray) -> str:
    """Returns 'palm_leaf' or 'stone' based on HSV hue and saturation."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_hue = float(hsv[:, :, 0].mean())
    mean_sat = float(hsv[:, :, 1].mean())
    if mean_sat > 40 and 8 <= mean_hue <= 30:
        return "palm_leaf"
    return "stone"


# ─── public dispatcher ────────────────────────────────────────────────────────

_METHODS = ("sauvola", "otsu", "adaptive", "unet", "docentr")


def binarise(
    img_path: str,
    output_path: str,
    method: str = "sauvola",
) -> np.ndarray:
    """Binarise image. Output is always white text on black background.
    method: 'sauvola' | 'otsu' | 'adaptive' | 'unet' | 'docentr'
    """
    # Use imdecode to handle paths with special characters on Windows
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    doc_type = detect_document_type(img)

    if method == "sauvola":
        if doc_type == "palm_leaf":
            binary = binarise_palm_leaf(img)
        else:
            binary = binarise_stone(img)
    elif method == "otsu":
        binary = binarise_otsu(img)
    elif method == "adaptive":
        binary = binarise_adaptive(img)
    elif method == "unet":
        binary = binarise_unet(img)
    elif method == "docentr":
        binary = binarise_docentr(img)
    else:
        raise ValueError(f"Unknown method '{method}'. Use: {' | '.join(_METHODS)}")

    # Safety: enforce white text on black background
    # If mean >= 127 the image is mostly white (wrong polarity) — flip it
    if binary.mean() >= 127:
        binary = cv2.bitwise_not(binary)

    # Noise removal with document-type appropriate parameters
    if doc_type == "palm_leaf" and method == "sauvola":
        binary = remove_noise_blobs(binary, min_size=8, min_length=15)
    else:
        binary = remove_noise_blobs(binary, min_size=80, min_length=25)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), binary)

    LOGGER.info(
        "Binarised %s -> %s (method=%s, doc_type=%s)",
        img_path, out, method, doc_type
    )
    return binary


def build_output_path(input_path: Path, output_dir: Path) -> Path:
    return Path(output_dir) / f"{Path(input_path).stem}_binarised.png"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3: Binarise inscription images")
    sub = parser.add_subparsers(dest="mode", required=True)

    single = sub.add_parser("single", help="Process one image")
    single.add_argument("input", help="Input image path")
    single.add_argument("output", help="Output PNG path")
    single.add_argument("--method", choices=_METHODS, default="sauvola")

    batch = sub.add_parser("batch", help="Process a directory")
    batch.add_argument("input_dir", help="Directory of images")
    batch.add_argument("output_dir", help="Directory for output PNGs")
    batch.add_argument("--method", choices=_METHODS, default="sauvola")
    batch.add_argument("--pattern", default="*.jpg", help="Glob pattern")

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.mode == "single":
        binarise(args.input, args.output, method=args.method)
    else:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        images = list(input_dir.glob(args.pattern))
        LOGGER.info("Found %d images in %s", len(images), input_dir)
        for img_path in images:
            out_path = build_output_path(img_path, output_dir)
            try:
                binarise(str(img_path), str(out_path), method=args.method)
            except Exception as exc:
                LOGGER.error("Failed %s: %s", img_path, exc)


if __name__ == "__main__":
    main()