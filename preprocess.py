"""
preprocess.py

Shared line-image preprocessing for training and inference.

`preprocess_line` picks a high-contrast channel for coloured ink, flattens
illumination, optionally Sauvola-binarizes, crops to the ink, deskews, and
resizes to height 64 without stretching. Callers pad on the right up to
`max_w` unless the line is so wide that fitting would squash the strokes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    """Options for `preprocess_line`. `channel` is auto, gray, min, green, or lab_l."""

    target_h: int = 64
    max_w: int = 1280
    sauvola: bool = False
    clahe: bool = True
    deskew: bool = True
    crop: bool = True
    channel: str = "auto"
    allow_wide: bool = True
    min_stroke_h: int = 32
    margin: int = 8


def _as_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _contrast(channel: np.ndarray) -> float:
    """Gap between the paper (bright) and the ink (dark)."""
    flat = channel.astype(np.float32).ravel()
    if flat.size == 0:
        return 0.0
    dark = float(np.percentile(flat, 5))
    bright = float(np.percentile(flat, 95))
    return bright - dark


def select_ink_channel(img: np.ndarray, which: str = "auto") -> np.ndarray:
    """Return one uint8 channel. `auto` keeps the candidate with the strongest ink."""
    if img.ndim == 2:
        gray = img
        if which in ("auto", "gray"):
            return gray
        bgr = _as_bgr(img)
    else:
        bgr = _as_bgr(img)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    lab_l = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    candidates = {
        "gray": gray,
        "min": np.min(bgr, axis=2),
        "green": bgr[:, :, 1],
        "lab_l": lab_l,
    }
    if which != "auto":
        if which not in candidates:
            raise ValueError(f"Unknown channel {which!r}. Use auto, gray, min, green, or lab_l.")
        return candidates[which]
    name = max(candidates, key=lambda key: _contrast(candidates[key]))
    logger.debug("Ink channel %s contrast %.1f", name, _contrast(candidates[name]))
    return candidates[name]


def flatten_background(img: np.ndarray) -> np.ndarray:
    """Divide out a large-kernel illumination estimate."""
    bg = cv2.GaussianBlur(img, (0, 0), sigmaX=21, sigmaY=21)
    bg = np.maximum(bg, 1)
    flat = cv2.divide(img, bg, scale=255)
    return np.clip(flat, 0, 255).astype(np.uint8)


def apply_clahe(img: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def sauvola_threshold(img: np.ndarray, window: int = 25, k: float = 0.2, r: float = 128.0) -> np.ndarray:
    """Local Sauvola threshold. Ink stays black (0) on a white (255) page."""
    if window % 2 == 0:
        window += 1
    values = img.astype(np.float32)
    mean = cv2.boxFilter(values, -1, (window, window), normalize=True)
    mean_sq = cv2.boxFilter(values * values, -1, (window, window), normalize=True)
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0))
    thresh = mean * (1.0 + k * (std / r - 1.0))
    binary = np.where(values > thresh, 255, 0).astype(np.uint8)
    return binary


def _ink_mask(img: np.ndarray) -> np.ndarray:
    """Foreground mask with specks removed. Ink is nonzero."""
    if img.std() < 1:
        return np.zeros(img.shape, dtype=np.uint8)
    _thr, inv = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(inv, connectivity=8)
    if count <= 1:
        return np.zeros(img.shape, dtype=np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    min_area = max(12, int(0.0015 * img.size))
    keep = np.zeros(count, dtype=np.uint8)
    keep[1:] = (areas >= min_area).astype(np.uint8)
    return (keep[labels] * 255).astype(np.uint8)


def crop_to_ink(img: np.ndarray, margin: int = 8) -> np.ndarray:
    """Tight crop around ink, ignoring isolated specks. Returns `img` if no ink is found."""
    mask = _ink_mask(img)
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return img
    y0 = max(int(ys.min()) - margin, 0)
    y1 = min(int(ys.max()) + margin + 1, img.shape[0])
    x0 = max(int(xs.min()) - margin, 0)
    x1 = min(int(xs.max()) + margin + 1, img.shape[1])
    if y1 - y0 < 2 or x1 - x0 < 2:
        return img
    return img[y0:y1, x0:x1]


def deskew_ink(img: np.ndarray, max_angle: float = 12.0) -> np.ndarray:
    """Straighten with minAreaRect. Near-level and extreme angles are left alone."""
    mask = _ink_mask(img)
    ys, xs = np.where(mask > 0)
    if xs.size < 40:
        ink = np.column_stack(np.where(img < 190))
        if ink.shape[0] < 40:
            return img
        coords = np.column_stack([ink[:, 1], ink[:, 0]]).astype(np.float32)
    else:
        coords = np.column_stack([xs, ys]).astype(np.float32)
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.4 or abs(angle) > max_angle:
        return img
    height, width = img.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(img, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=255)


def resize_pad(img: np.ndarray, target_h: int, max_w: int, allow_wide: bool, min_stroke_h: int) -> np.ndarray:
    """Resize to `target_h` keeping aspect ratio, then right-pad. Never stretch."""
    height, width = img.shape
    if height < 1 or width < 1:
        canvas = np.full((target_h, max_w), 255, dtype=np.uint8)
        return canvas
    natural_w = max(1, int(round(width * (target_h / float(height)))))
    if natural_w <= max_w:
        resized = cv2.resize(img, (natural_w, target_h), interpolation=cv2.INTER_AREA)
        canvas = np.full((target_h, max_w), 255, dtype=np.uint8)
        canvas[:, :natural_w] = resized
        return canvas

    # Fitting the width would shrink the height. Refuse when strokes would vanish.
    fitted_h = max(1, int(round(height * (max_w / float(width)))))
    if allow_wide and fitted_h < min_stroke_h:
        logger.info(
            "Wide line kept at %d x %d (fitting to width %d would shrink height to %d).",
            target_h,
            natural_w,
            max_w,
            fitted_h,
        )
        resized = cv2.resize(img, (natural_w, target_h), interpolation=cv2.INTER_AREA)
        return resized

    scale = max_w / float(width)
    new_h = max(1, int(round(height * scale)))
    resized = cv2.resize(img, (max_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, max_w), 255, dtype=np.uint8)
    top = max((target_h - new_h) // 2, 0)
    canvas[top : top + new_h, :] = resized[: target_h - top, :]
    return canvas


def preprocess_line(img: np.ndarray, config: PreprocessConfig | None = None) -> np.ndarray:
    """
    Prepare one handwritten line.

    Returns a uint8 grayscale image of height `config.target_h`. Width is
    `config.max_w` with right padding, or wider when preserving the aspect
    ratio is required to keep the strokes readable.
    """
    if img is None or np.size(img) == 0:
        raise ValueError("Empty image passed to preprocess_line")
    cfg = config or PreprocessConfig()
    gray = select_ink_channel(img, cfg.channel)
    gray = flatten_background(gray)
    if cfg.clahe:
        gray = apply_clahe(gray)
    if cfg.sauvola:
        gray = sauvola_threshold(gray)
    if cfg.crop:
        gray = crop_to_ink(gray, margin=cfg.margin)
    if cfg.deskew:
        gray = deskew_ink(gray)
    return resize_pad(gray, cfg.target_h, cfg.max_w, cfg.allow_wide, cfg.min_stroke_h)
