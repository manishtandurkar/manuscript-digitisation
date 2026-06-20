from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

try:
    from src import binarise as core
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import binarise as core


CANVAS = 48
MEDIUM_MAX = 3000
BORDER_MARGIN = 3


def safe_imread(path: str) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def normalise_glyph(crop: np.ndarray, canvas: int = CANVAS) -> np.ndarray:
    h, w = crop.shape
    scale = (canvas - 4) / max(h, w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((canvas, canvas), dtype=np.uint8)
    y0 = (canvas - nh) // 2
    x0 = (canvas - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resized
    return out


def smooth_glyph(mask: np.ndarray) -> np.ndarray:
    k = np.ones((3, 3), np.uint8)
    smoothed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, k)
    blurred = cv2.GaussianBlur(smoothed, (3, 3), 0)
    _, smoothed = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY)
    return smoothed


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def cluster_glyphs(glyphs, threshold: float):
    clusters = []
    representatives = []
    for idx, g in enumerate(glyphs):
        best_score = 0.0
        best_cluster = -1
        for ci, rep in enumerate(representatives):
            score = iou(g, rep)
            if score > best_score:
                best_score = score
                best_cluster = ci
        if best_score >= threshold:
            clusters[best_cluster].append(idx)
            members = [glyphs[i] for i in clusters[best_cluster]]
            stacked = np.stack(members, axis=0).astype(np.float32) / 255.0
            avg = stacked.mean(axis=0)
            representatives[best_cluster] = (avg > 0.5).astype(np.uint8) * 255
        else:
            clusters.append([idx])
            representatives.append(g.copy())
    return clusters, representatives


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--out-dir", default="./cluster_output")
    parser.add_argument("--iou-threshold", type=float, default=0.45)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)
    (out_dir / "clusters").mkdir(parents=True, exist_ok=True)

    img = safe_imread(args.image)
    h_img, w_img = img.shape[:2]

    binary = core.binarise_stone(img)
    if binary.mean() >= 127:
        binary = cv2.bitwise_not(binary)
    binary = core.remove_noise_blobs(binary, min_size=80, min_length=25)
    cv2.imwrite(str(out_dir / "binarised.png"), binary)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    valid_indices = []
    flagged_merged = []
    excluded_artifacts = []

    for i in range(1, n):
        x, y, w, h, area = stats[i]
        touches_border = (x <= BORDER_MARGIN or y <= BORDER_MARGIN or
                           x + w >= w_img - BORDER_MARGIN or y + h >= h_img - BORDER_MARGIN)
        if area >= MEDIUM_MAX:
            if touches_border:
                excluded_artifacts.append(i)
            else:
                flagged_merged.append(i)
        else:
            valid_indices.append(i)

    print(f"[components] total={n - 1}  glyph_candidates={len(valid_indices)}  "
          f"flagged_merged(manual review)={len(flagged_merged)}  "
          f"excluded_border_artifacts={len(excluded_artifacts)}")

    glyph_masks = []
    glyph_boxes = []
    for i in valid_indices:
        x, y, w, h, area = stats[i]
        crop = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
        glyph_masks.append(normalise_glyph(crop))
        glyph_boxes.append((x, y, w, h))
        cv2.imwrite(str(out_dir / "crops" / f"glyph_{i:04d}.png"), crop)

    reconstructed = np.zeros_like(binary)
    for idx, (x, y, w, h) in enumerate(glyph_boxes):
        i = valid_indices[idx]
        raw_crop = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
        cleaned = smooth_glyph(raw_crop)
        roi = reconstructed[y:y + h, x:x + w]
        reconstructed[y:y + h, x:x + w] = np.maximum(roi, cleaned)

    strict_threshold = max(args.iou_threshold, 0.70)
    clusters, representatives = cluster_glyphs(glyph_masks, strict_threshold)
    multi_member = [c for c in clusters if len(c) >= 2]
    print(f"\n[repeat-detection, threshold={strict_threshold}] {len(glyph_masks)} glyphs -> "
          f"{len(clusters)} groups, {len(multi_member)} groups with 2+ members "
          f"(UNVERIFIED — inspect clusters/ manually)")

    for ci, (members, rep) in enumerate(zip(clusters, representatives)):
        if len(members) >= 2:
            cv2.imwrite(str(out_dir / "clusters" / f"cluster_{ci:03d}_n{len(members)}.png"), rep)

    for i in flagged_merged:
        x, y, w, h, area = stats[i]
        mask = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
        roi = reconstructed[y:y + h, x:x + w]
        reconstructed[y:y + h, x:x + w] = np.maximum(roi, mask)

    cv2.imwrite(str(out_dir / "reconstructed_clean.png"), reconstructed)

    print(f"\n[output] binarised image -> {out_dir / 'binarised.png'}")
    print(f"[output] per-glyph crops -> {out_dir / 'crops'}/")
    print(f"[output] cluster representatives -> {out_dir / 'clusters'}/")
    print(f"[output] reconstructed clean image -> {out_dir / 'reconstructed_clean.png'}")
    print(f"\n[manual review needed] {len(flagged_merged)} merged-character blobs left unmodified "
          f"at component ids: {flagged_merged}")


if __name__ == "__main__":
    main()
