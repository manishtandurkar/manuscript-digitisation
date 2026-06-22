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
    Stone inscription binarisation via Frangi vesselness.

    Handles two polarities automatically:
    - Dark background / light text (typical carved stone):
        white-hat lifts bright strokes, Frangi detects them.
    - Light background / dark text (outdoor photograph, weathered stone):
        CLAHE + invert + vegetation mask (low-saturation pixels) + Frangi
        detects inverted dark strokes.

    Small images (shorter < 400px) are upscaled 3× so thin strokes survive
    the outer noise-removal pass in binarise() which uses min_length=25.
    Output: white text, black background.
    """
    try:
        from skimage.filters import frangi
    except ImportError:
        LOGGER.warning("scikit-image frangi not available — falling back to Sauvola")
        return binarise_sauvola(img)

    gray = _to_gray(img)
    H0, W0 = gray.shape
    shorter0 = min(H0, W0)
    gray_mean = float(gray.mean())
    gray_std  = float(gray.std())

    # ── Polarity detection ───────────────────────────────────────────────────
    # mean > 128 → bright stone background, dark carved text → invert path
    bright_bg = gray_mean > 128

    if bright_bg:
        # Outdoor / bright-stone path: mask vegetation, invert, Frangi on gray
        scale = 3 if shorter0 < 400 else (2 if shorter0 < 800 else 1)
        if scale > 1:
            gray_work = cv2.resize(gray, (W0 * scale, H0 * scale),
                                   interpolation=cv2.INTER_LANCZOS4)
            img_work = cv2.resize(img, (W0 * scale, H0 * scale),
                                  interpolation=cv2.INTER_LANCZOS4) if img.ndim == 3 else gray_work
        else:
            gray_work = gray
            img_work = img
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray_work)
        # Stone isolation: background is dark (val<80) OR highly saturated (sat>80,
        # i.e. green vegetation). Find the largest connected non-background region.
        if img_work.ndim == 3:
            hsv = cv2.cvtColor(img_work, cv2.COLOR_BGR2HSV)
            sat_ch, val_ch = hsv[:, :, 1], hsv[:, :, 2]
            bg = ((val_ch < 80) | (sat_ch > 80)).astype(np.uint8) * 255
            bg = cv2.morphologyEx(bg, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
            not_bg = cv2.bitwise_not(bg)
            not_bg = cv2.morphologyEx(not_bg, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
            n_cc, lbl, cc_stats, _ = cv2.connectedComponentsWithStats(not_bg, connectivity=8)
            stone_mask = np.zeros(gray_eq.shape, dtype=np.uint8)
            min_stone = int(gray_work.shape[0] * gray_work.shape[1] * 0.05)
            for ci in range(1, n_cc):
                if cc_stats[ci, cv2.CC_STAT_AREA] >= min_stone:
                    stone_mask[lbl == ci] = 255
            stone_mask = cv2.dilate(stone_mask, np.ones((20, 20), np.uint8))
        else:
            stone_mask = np.full(gray_eq.shape, 255, dtype=np.uint8)
        gray_inv = cv2.bitwise_not(gray_eq)
        gray_inv[stone_mask == 0] = 0
        frangi_input = gray_inv.astype(np.float32) / 255.0
        frangi_sigmas = range(2, 10)
        gray_up = gray_inv
    else:
        # Dark-stone path: upscale small images, white-hat, Frangi
        scale = 3 if shorter0 < 400 else (2 if shorter0 < 800 else 1)
        if scale > 1:
            gray_up = cv2.resize(gray, (W0 * scale, H0 * scale),
                                 interpolation=cv2.INTER_LANCZOS4)
        else:
            gray_up = gray
        H, W = gray_up.shape
        shorter = min(H, W)
        wh_k = max(21, shorter // 6)
        if wh_k % 2 == 0:
            wh_k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (wh_k, wh_k))
        wh = cv2.morphologyEx(gray_up, cv2.MORPH_TOPHAT, kernel)
        wh = cv2.normalize(wh, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        frangi_input = wh.astype(np.float32) / 255.0
        frangi_sigmas = range(1, 7)
        stone_mask = None

    H, W = gray_up.shape
    shorter = min(H, W)

    # ── Frangi vesselness ────────────────────────────────────────────────────
    vessel = frangi(frangi_input, sigmas=frangi_sigmas, black_ridges=False)
    vn = cv2.normalize(vessel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vn = cv2.medianBlur(vn, 3)

    # Adaptive percentile: low-contrast → p83; high-contrast → p88
    t_pct = int(np.clip(88 - max(0.0, (65 - gray_std) * 0.25), 83, 88))
    # Compute percentile only over stone pixels to avoid zero-padding bias
    if stone_mask is not None:
        vn_stone = vn[stone_mask > 0]
        t_frangi = int(np.percentile(vn_stone, t_pct)) if vn_stone.size else 0
        vn[stone_mask == 0] = 0
    else:
        t_frangi = int(np.percentile(vn, t_pct))
    LOGGER.debug("binarise_stone: mean=%.1f std=%.1f bright_bg=%s t_pct=%d t=%d",
                 gray_mean, gray_std, bright_bg, t_pct, t_frangi)
    binary = (vn > t_frangi).astype(np.uint8) * 255
    if stone_mask is not None:
        binary[stone_mask == 0] = 0

    # ── Morphological cleanup ────────────────────────────────────────────────
    close_k = max(3, shorter // 150)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                              np.ones((close_k, close_k), np.uint8))
    min_size = max(12, (shorter // 80) ** 2)
    binary = remove_noise_blobs(binary,
                                min_size=min_size,
                                min_length=max(5, shorter // 60))
    # Base merge_k on original (pre-upscale) shorter side so it stays proportional
    # regardless of whether upscaling was applied.
    merge_k = max(3, shorter0 // 120)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                              np.ones((merge_k, merge_k), np.uint8))

    if binary.mean() >= 127:
        binary = cv2.bitwise_not(binary)
    return binary

def _palm_leaf_rough_mask(img: np.ndarray) -> np.ndarray:
    """
    Fast rough binary mask used to locate character regions on palm-leaf images.
    R-channel + bilateral + CLAHE + Sauvola. Not the final output — used only
    as a segmentation guide for binarise_palm_leaf.
    """
    from skimage.filters import threshold_sauvola

    h, w = img.shape[:2]
    shorter = min(h, w)
    r = img[:, :, 2]
    sigma_s = max(5, shorter // 30)
    denoised = cv2.bilateralFilter(r, d=9, sigmaColor=30, sigmaSpace=sigma_s)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)
    ws = max(21, min(61, (shorter // 8) | 1))
    if ws % 2 == 0:
        ws += 1
    thresh = threshold_sauvola(enhanced, window_size=ws, k=0.18)
    binary = (enhanced < thresh).astype(np.uint8) * 255
    mask = np.zeros((h + 2, w + 2), np.uint8)
    for corner in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
        cv2.floodFill(binary, mask, (corner[1], corner[0]), 0)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    min_size = max(15, (shorter // 120) ** 2)
    return remove_noise_blobs(binary, min_size=min_size, min_length=max(8, shorter // 80))


def binarise_palm_leaf(img: np.ndarray) -> np.ndarray:
    """
    Palm-leaf manuscript binarisation via character-level segmentation.

    1. Produce a rough binary mask (R-channel + bilateral + Sauvola) to locate
       character regions.
    2. Dilate mask to merge nearby strokes into whole-character blobs.
    3. Find connected components — each is one character or ligature cluster.
    4. For each component: crop the same region from the original colour image,
       apply a tight local Sauvola threshold on the R-channel of that crop.
    5. Stamp each locally-binarised crop onto a black canvas.

    Per-character local thresholding eliminates background bleed that global
    methods miss, since each small crop has near-uniform local illumination.
    Output: white text, black background.
    """
    from skimage.filters import threshold_sauvola

    H, W = img.shape[:2]
    shorter = min(H, W)

    # Step 1: rough mask for segmentation guidance
    rough = _palm_leaf_rough_mask(img)

    # Step 2: dilate to merge intra-character stroke gaps
    dil_k = max(3, shorter // 40)
    dilated = cv2.dilate(rough, np.ones((dil_k, dil_k), np.uint8))

    # Step 3: connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)

    min_area = max(20, (shorter // 60) ** 2)
    max_area = int(H * W * 0.40)
    pad = max(2, shorter // 80)
    canvas = np.zeros((H, W), dtype=np.uint8)

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue

        x  = int(stats[label, cv2.CC_STAT_LEFT])
        y  = int(stats[label, cv2.CC_STAT_TOP])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])

        x0 = max(0, x - pad);  y0 = max(0, y - pad)
        x1 = min(W, x + cw + pad); y1 = min(H, y + ch + pad)

        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            continue

        # Step 4: local binarisation on the R-channel crop
        r_crop = crop[:, :, 2]
        cr_h, cr_w = r_crop.shape
        if cr_h < 10 or cr_w < 10:
            _, local_bin = cv2.threshold(r_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            ws = max(7, min(31, (min(cr_h, cr_w) // 3) | 1))
            if ws % 2 == 0:
                ws += 1
            thresh = threshold_sauvola(r_crop, window_size=ws, k=0.15)
            local_bin = (r_crop < thresh).astype(np.uint8) * 255

        # Step 5: stamp onto canvas
        canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], local_bin)

    # Final cleanup
    canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    canvas = remove_noise_blobs(
        canvas,
        min_size=max(10, (shorter // 150) ** 2),
        min_length=max(5, shorter // 100),
    )
    if canvas.mean() >= 127:
        canvas = cv2.bitwise_not(canvas)
    return canvas


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


def detect_rubbing(img: np.ndarray) -> bool:
    """Detect rubbing/estampage-style stone images (chalk-on-ink, paper grain noise).

    Signature: moderate-to-high saturation (ink residue), high local speckle
    texture (paper grain), and strong global contrast (near-bimodal histogram).
    Thresholds derived empirically from diagnostic testing.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_sat = float(hsv[:, :, 1].mean())

    gray = _to_gray(img).astype(np.float32)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    local_var = cv2.GaussianBlur((gray - blurred) ** 2, (15, 15), 0)
    mean_local_std = float(local_var.mean()) ** 0.5

    global_std = float(gray.std())

    return mean_sat > 35 and mean_local_std > 18 and global_std > 50


# ─── public dispatcher ────────────────────────────────────────────────────────

_METHODS = ("sauvola", "otsu", "adaptive", "unet", "docentr")




def binarise_rubbing(img: np.ndarray) -> np.ndarray:
    """Dedicated path for rubbings/estampages. Median-blur + Otsu only.

    Parameters (blur=13, open=2, close=2) determined empirically from
    diagnostic grid-search testing on rubbing-style stone inscriptions.
    Output: white text, black background.
    """
    gray = _to_gray(img)
    median = cv2.medianBlur(gray, 13)
    _, binary = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    if binary.mean() >= 127:
        binary = cv2.bitwise_not(binary)
    return binary
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
        elif detect_rubbing(img):
            binary = binarise_rubbing(img)
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
