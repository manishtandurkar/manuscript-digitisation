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
        Train on THPLMD grayscale→binary pairs (input: 1×H×W float, output: 1×H×W sigmoid).
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
            x = x.unfold(2, ps, ps).unfold(3, ps, ps)  # B×1×hp×wp×ps×ps
            hp, wp = x.shape[2], x.shape[3]
            B = x.shape[0]
            x = x.contiguous().view(B, hp * wp, ps * ps)
            return self.proj(x), hp, wp

    class _DocEnTr(nn.Module):
        """
        Simplified DocEnTr: patch-ViT encoder + CNN decoder for binarisation.
        Ref: El-Hajj & Barakat, ArXiv 2209.09921.
        Train on THPLMD grayscale→binary pairs (input: 1×H×W float, output: 1×H×W sigmoid).
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
    """Returns (prob_map H×W float32, confidence). prob_map None if model unavailable."""
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


def _sauvola_window(gray: np.ndarray) -> int:
    """Window ~1/20 of shorter dimension, clamped to [15, 51], always odd."""
    w = max(15, min(gray.shape[0], gray.shape[1]) // 20)
    return w if w % 2 == 1 else w + 1


def binarise_sauvola(img: np.ndarray, window_size: int | None = None) -> np.ndarray:
    """Sauvola local thresholding — best for uneven backgrounds (stone, palm leaf)."""
    from skimage.filters import threshold_sauvola

    gray = _clahe(_to_gray(img))
    ws = window_size if window_size is not None else _sauvola_window(gray)
    thresh = threshold_sauvola(gray, window_size=ws)
    binary = (gray > thresh).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def binarise_otsu(img: np.ndarray) -> np.ndarray:
    """Otsu global thresholding — fast, good for clean paper manuscripts."""
    gray = _clahe(_to_gray(img))
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def binarise_adaptive(img: np.ndarray) -> np.ndarray:
    """OpenCV adaptive mean thresholding — fallback for mixed quality images."""
    gray = _clahe(_to_gray(img))
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 8
    )
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def binarise_stone(img: np.ndarray) -> np.ndarray:
    """Specialised stone inscription binarisation.

    Stone grain (1–5 px) and carved text strokes (8–30 px) differ in scale.
    Pipeline:
      1. Gaussian pre-smooth (sigma=3) — kills sub-5px grain without affecting strokes.
      2. Black-hat morphological transform — detects dark recesses (carvings) relative
         to a structuring element sized to expected stroke width; suppresses
         large-scale background variation at the same time.
      3. Normalize + Otsu — bimodal distribution (stone≈0, carvings>0) after black-hat.
      4. Morphological clean-up — open removes residual grain speckles, close fills gaps.
    Output: white text on black background.
    """
    gray = _to_gray(img)
    h, w = gray.shape

    # 1. Stronger pre-smooth: sigma=5 kills <10px grain, carved grooves (15-50px) survive
    smooth = cv2.GaussianBlur(gray, (0, 0), sigmaX=5, sigmaY=5)

    # 2. Black-hat with kernel sized to span full stroke width (~1/12 of short edge).
    #    Grooves narrower than k appear bright; wider grooves + flat background = 0.
    k = max(31, min(h, w) // 12)
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    black_hat = cv2.morphologyEx(smooth, cv2.MORPH_BLACKHAT, kernel)

    # 3. Normalize then keep only top-30% response (carved grooves dominate the peak;
    #    grain texture sits in the lower tail — percentile threshold is more selective
    #    than Otsu when the distribution is not cleanly bimodal).
    black_hat = cv2.normalize(black_hat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    thresh_val = int(np.percentile(black_hat, 75))
    thresh_val = max(thresh_val, 30)  # floor: never threshold below 30
    _, binary = cv2.threshold(black_hat, thresh_val, 255, cv2.THRESH_BINARY)

    # 4. Open (3×3) removes surviving grain speckles; close (5×5) fills stroke gaps
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    return binary


def binarise_palm_leaf(img: np.ndarray) -> np.ndarray:
    """Specialised palm-leaf manuscript binarisation path.

    Uses LAB-space L-channel Sauvola with a widened window and an
    A-channel Otsu mask to suppress warm-background fibre texture.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 5
    )
    binary = cv2.bitwise_not(binary)

    h, w = binary.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    for corner in [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]:
        cv2.floodFill(binary, mask, (corner[1], corner[0]), 0)

    kernel = np.ones((2, 2), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary


# ─── DL methods (public) ─────────────────────────────────────────────────────

def binarise_unet(
    img: np.ndarray,
    weights_path: Path | None = None,
) -> np.ndarray:
    """
    Lightweight U-Net binarisation.
    Falls back to Sauvola when confidence < _CONFIDENCE_THRESHOLD or weights absent.
    Train: THPLMD grayscale input → binary ground-truth pairs.
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
    Train: THPLMD grayscale input → binary ground-truth pairs.
    """
    wp = Path(weights_path) if weights_path else _MODELS_DIR / "docentr_binarise.pth"
    prob, conf = _dl_infer(img, "docentr", wp)
    if prob is None or conf < _CONFIDENCE_THRESHOLD:
        LOGGER.info("docentr confidence %.3f below threshold — Sauvola fallback", conf)
        return binarise_sauvola(img)
    return _prob_to_binary(prob)


# ─── noise removal ────────────────────────────────────────────────────────────

def remove_noise_blobs(
    binary: np.ndarray, min_size: int = 50, min_length: int = 30
) -> np.ndarray:
    """Remove small disconnected components (dust, noise) from binary image.

    Preserves components that are long/large even if their area is small
    (useful for thin strokes on palm leaves). Keeps a component if
    area >= min_size OR max(width, height) >= min_length.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    cleaned = np.zeros_like(binary)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area >= min_size or max(w, h) >= min_length:
            cleaned[labels == label] = 255
    return cleaned


# ---------------------------------------------------------------------------
# Document-type detection and palm-leaf binarisation
# Palm leaf manuscripts have warm orange/tan backgrounds (high saturation,
# hue 8–30 in OpenCV HSV). The standard CLAHE+Sauvola pipeline (designed
# for stone inscriptions with near-achromatic backgrounds) fails on these
# because it boosts fibre texture as aggressively as ink strokes.
# The palm-leaf path uses LAB space to separate L (lightness) and A
# (green-red axis) signals, applies mild CLAHE only to L, widens the
# Sauvola window to average over fibre texture, and intersects the result
# with an A-channel Otsu mask to eliminate warm-background false positives.
# The stone/grayscale path is completely unchanged.
# ---------------------------------------------------------------------------


def detect_document_type(img: np.ndarray) -> str:
    """Heuristic document-type detector returning 'palm_leaf' or 'stone'.

    Uses mean HSV hue and saturation to detect warm, tan palm-leaf backgrounds.
    """
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
    """Binarise image. method: 'sauvola' | 'otsu' | 'adaptive' | 'unet' | 'docentr'"""
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    # document-type aware routing for default 'sauvola' method
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

    if doc_type == "palm_leaf" and method == "sauvola":
        binary = remove_noise_blobs(binary, min_size=8, min_length=15)
    elif doc_type == "stone":
        binary = remove_noise_blobs(binary, min_size=200, min_length=30)
    else:
        binary = remove_noise_blobs(binary)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), binary)

    LOGGER.info(
        "Binarised %s → %s (method=%s, doc_type=%s)",
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
