"""
segment_lines.py

Detect and crop handwritten text lines from a page image.

Primary: YOLOv8 line detector (checkpoints/line_detector.pt).
Fallback: ink projection + OPTICS.

Usage:
    python segment_lines.py --image path\\to\\full_page.jpg --out_dir segmented_lines
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import OPTICS

_ROOT = Path(__file__).resolve().parent
_DEFAULT_DETECTOR = _ROOT / "checkpoints" / "line_detector.pt"

_yolo_model = None
_yolo_path = None


def _binarize(img):
    """Return ink=255, background=0 binary image."""
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.dilate(opened, kernel, iterations=1)


def _estimate_line_pitch(smooth):
    n = len(smooth)
    if n < 40 or smooth.max() <= 0:
        return None

    x = smooth - float(smooth.mean())
    fft = np.fft.rfft(x, n=2 * n)
    ac = np.fft.irfft(fft * np.conj(fft))[:n]
    if ac[0] <= 1e-8:
        return None
    ac = ac / ac[0]

    lo = max(18, n // 100)
    hi = max(lo + 5, min(n // 3, 220))
    region = ac[lo:hi]
    if region.size == 0:
        return None

    peak_rel = int(np.argmax(region))
    pitch = lo + peak_rel
    if ac[pitch] < 0.12:
        return None
    return int(pitch)


def _find_peaks(smooth, min_distance, prominence_ratio=0.08):
    n = len(smooth)
    if n == 0 or smooth.max() <= 0:
        return []

    thr = max(smooth.max() * prominence_ratio, float(np.percentile(smooth, 60)) * 0.35)
    candidates = []
    for i in range(1, n - 1):
        if smooth[i] >= thr and smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[i + 1]:
            candidates.append(i)

    if not candidates:
        return []

    candidates = sorted(candidates, key=lambda i: smooth[i], reverse=True)
    peaks = []
    for i in candidates:
        if all(abs(i - p) >= min_distance for p in peaks):
            peaks.append(i)
    return sorted(peaks)


def _merge_close_peaks(peaks, smooth, min_sep):
    if len(peaks) < 2:
        return peaks
    keep = [peaks[0]]
    for p in peaks[1:]:
        if p - keep[-1] < min_sep:
            if smooth[p] > smooth[keep[-1]]:
                keep[-1] = p
        else:
            keep.append(p)
    return keep


def _peak_bands(smooth, peaks, page_h):
    if not peaks:
        return []

    bounds = [0]
    for a, b in zip(peaks[:-1], peaks[1:]):
        valley = a + int(np.argmin(smooth[a : b + 1]))
        bounds.append(valley)
    bounds.append(page_h - 1)

    bands = []
    for i, peak in enumerate(peaks):
        top, bottom = bounds[i], bounds[i + 1]
        floor = max(smooth[peak] * 0.18, smooth.max() * 0.04)
        y0 = peak
        while y0 > top and smooth[y0] >= floor:
            y0 -= 1
        y1 = peak
        while y1 < bottom and smooth[y1] >= floor:
            y1 += 1
        if y1 - y0 < 8:
            y0 = max(0, peak - 10)
            y1 = min(page_h, peak + 10)
        bands.append((int(y0), int(y1)))
    return bands


def _trim_horizontal(crop, pad=8):
    if crop.size == 0:
        return crop
    ink_cols = np.where(np.any(crop < 200, axis=0))[0]
    if ink_cols.size == 0:
        return crop
    x0 = max(0, int(ink_cols[0]) - pad)
    x1 = min(crop.shape[1], int(ink_cols[-1]) + 1 + pad)
    return crop[:, x0:x1]


def _bands_to_crops(img, bands, pad_ratio=0.12):
    crops = []
    page_h = img.shape[0]
    for y_top, y_bottom in bands:
        h = max(1, y_bottom - y_top)
        pad = max(3, int(h * pad_ratio))
        y0 = max(0, y_top - pad)
        y1 = min(page_h, y_bottom + pad)
        crop = img[y0:y1, :]
        if crop.size == 0:
            continue
        crop = _trim_horizontal(crop)
        ink = np.count_nonzero(crop < 200)
        min_ink = max(60, crop.shape[0] * 2)
        if ink < min_ink:
            continue
        crops.append(crop)
    return crops


def _segment_by_projection(img, binary):
    page_h = img.shape[0]
    row_ink = (binary > 0).sum(axis=1).astype(np.float32)

    k = max(5, (int(page_h * 0.006) | 1))
    smooth = np.convolve(row_ink, np.ones(k, dtype=np.float32) / k, mode="same")

    pitch = _estimate_line_pitch(smooth)
    if pitch is None:
        min_distance = max(10, page_h // 50)
        merge_sep = max(12, page_h // 70)
    else:
        min_distance = max(8, int(0.55 * pitch))
        merge_sep = max(10, int(0.40 * pitch))

    peaks = _find_peaks(smooth, min_distance=min_distance, prominence_ratio=0.07)
    peaks = _merge_close_peaks(peaks, smooth, min_sep=merge_sep)

    if pitch is not None and peaks:
        expected = max(1, int(round(page_h / pitch)))
        if len(peaks) > expected * 1.6:
            peaks = _find_peaks(
                smooth, min_distance=max(8, int(0.7 * pitch)), prominence_ratio=0.09
            )
            peaks = _merge_close_peaks(peaks, smooth, min_sep=max(10, int(0.45 * pitch)))
        elif len(peaks) < max(2, expected * 0.45):
            peaks = _find_peaks(
                smooth, min_distance=max(6, int(0.4 * pitch)), prominence_ratio=0.05
            )
            peaks = _merge_close_peaks(peaks, smooth, min_sep=max(8, int(0.35 * pitch)))

    return _peak_bands(smooth, peaks, page_h)


def _segment_by_optics(img, binary, min_area=80):
    page_h, page_w = img.shape
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, h_kernel, iterations=1)

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
    boxes, midpoints = [], []
    min_h = max(6, int(0.008 * page_h))
    max_h = int(0.25 * page_h)
    min_w = max(8, int(0.01 * page_w))

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < min_area or h < min_h or h > max_h or w < min_w:
            continue
        boxes.append((x, y, w, h))
        midpoints.append(centroids[i])

    if len(midpoints) < 2:
        if not boxes:
            return []
        y_top = min(b[1] for b in boxes)
        y_bottom = max(b[1] + b[3] for b in boxes)
        return [(y_top, y_bottom)]

    y_coords = np.array([[m[1]] for m in midpoints])
    clustering = OPTICS(min_samples=3, xi=0.12, min_cluster_size=3).fit(y_coords)

    clusters = {}
    for box, label in zip(boxes, clustering.labels_):
        if label == -1:
            continue
        clusters.setdefault(label, []).append(box)

    bands = []
    for cluster_boxes in clusters.values():
        y_top = min(b[1] for b in cluster_boxes)
        y_bottom = max(b[1] + b[3] for b in cluster_boxes)
        bands.append((y_top, y_bottom))

    bands = sorted(bands, key=lambda b: b[0])
    if not bands:
        return []

    heights = [b[1] - b[0] for b in bands]
    med_h = float(np.median(heights))
    merged = [list(bands[0])]
    for top, bottom in bands[1:]:
        if top - merged[-1][1] <= 0.35 * med_h:
            merged[-1][1] = max(merged[-1][1], bottom)
        else:
            merged.append([top, bottom])
    return [(int(t), int(b)) for t, b in merged]


def _segment_classical(img, min_area=80):
    binary = _binarize(img)
    bands = _segment_by_projection(img, binary)
    if len(bands) < 1:
        bands = _segment_by_optics(img, binary, min_area=min_area)
    return _bands_to_crops(img, bands)


def _load_yolo(weights_path):
    global _yolo_model, _yolo_path
    path = str(Path(weights_path).resolve())
    if _yolo_model is not None and _yolo_path == path:
        return _yolo_model
    from ultralytics import YOLO

    _yolo_model = YOLO(path)
    _yolo_path = path
    return _yolo_model


def _boxes_to_crops(img, boxes_xyxy, confs=None, conf_thr=0.25, pad_ratio=0.04):
    """Crop line strips from YOLO boxes (full width, horizontal trim)."""
    page_h, page_w = img.shape[:2]
    items = []
    for i, box in enumerate(boxes_xyxy):
        conf = 1.0 if confs is None else float(confs[i])
        if conf < conf_thr:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        bh = max(1.0, y2 - y1)
        if bh < 4:
            continue
        cy = 0.5 * (y1 + y2)
        half = 0.46 * bh
        pad_y = max(2, int(bh * pad_ratio))
        ya = max(0, int(cy - half) - pad_y)
        yb = min(page_h, int(cy + half) + pad_y)
        strip = img[ya:yb, :]
        if strip.size == 0:
            continue
        crop = _trim_horizontal(strip, pad=10)
        ink = np.count_nonzero(crop < 200)
        if ink < max(40, crop.shape[0]):
            continue
        items.append((cy, crop))

    if not items:
        return []
    items.sort(key=lambda t: t[0])

    merged = [items[0]]
    for cy, crop in items[1:]:
        p_cy, p_crop = merged[-1]
        p_h = p_crop.shape[0]
        h = crop.shape[0]
        if abs(cy - p_cy) < 0.4 * max(p_h, h):
            if crop.shape[0] * crop.shape[1] > p_crop.shape[0] * p_crop.shape[1]:
                merged[-1] = (cy, crop)
        else:
            merged.append((cy, crop))
    return [c for _, c in merged]


def _segment_by_yolo(img, weights_path, conf=0.25, imgsz=1280, source=None):
    """Run YOLO detection and crop line images from grayscale `img`."""
    model = _load_yolo(weights_path)
    if source is None:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
        source = np.ascontiguousarray(bgr)
        if source.ndim != 3 or source.shape[2] != 3:
            raise ValueError(f"YOLO source must be HxWx3, got {getattr(source, 'shape', None)}")

    results = model.predict(source, conf=conf, imgsz=imgsz, verbose=False)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return []

    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    orig_shape = results[0].orig_shape
    page_h, page_w = img.shape[:2]
    if orig_shape is not None:
        oh, ow = int(orig_shape[0]), int(orig_shape[1])
        if oh != page_h or ow != page_w:
            sx = page_w / float(ow)
            sy = page_h / float(oh)
            boxes = boxes.copy()
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy

    return _boxes_to_crops(img, boxes, confs=confs, conf_thr=conf)


def looks_like_single_line(img):
    """Return True if the image looks like a single pre-cropped text line."""
    if img is None or img.size == 0:
        return False
    h, w = img.shape[:2]
    if h < 8 or w < 8:
        return False
    aspect = w / float(h)
    if aspect < 2.5:
        return False
    if h > 220 and aspect < 4.0:
        return False

    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = _binarize(gray)
    row_ink = (binary > 0).sum(axis=1).astype(np.float32)
    if row_ink.max() <= 0:
        return True
    k = max(3, (h // 20) | 1)
    smooth = np.convolve(row_ink, np.ones(k, dtype=np.float32) / k, mode="same")
    peaks = _find_peaks(smooth, min_distance=max(8, h // 4), prominence_ratio=0.15)
    return len(peaks) <= 1


def resolve_detector_weights(explicit=None):
    candidates = []
    if explicit:
        candidates.append(explicit)
    env = os.environ.get("LINE_DETECTOR")
    if env:
        candidates.append(env)
    candidates.append(str(_DEFAULT_DETECTOR))
    for path in candidates:
        if path and os.path.isfile(path):
            return os.path.abspath(path)
    return None


def segment_page(img_path, min_area=80, detector_weights=None, force_classical=False):
    """Detect and crop text lines. Returns (image, line_crops, method)."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(img_path)

    weights = None if force_classical else resolve_detector_weights(detector_weights)
    method = "classical"

    if weights is not None:
        try:
            line_images = _segment_by_yolo(img, weights, source=img_path)
            if line_images:
                return img, line_images, "yolo"
        except Exception as exc:
            print(f"[segment_lines] YOLO failed ({exc}); falling back to classical")

    target_w = 1200
    scale = target_w / img.shape[1]
    resized = cv2.resize(img, (target_w, int(img.shape[0] * scale)))
    line_images = _segment_classical(resized, min_area=min_area)
    return resized, line_images, method


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--detector", default=None, help="Path to line_detector.pt")
    ap.add_argument("--classical", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    _, lines, method = segment_page(
        args.image, detector_weights=args.detector, force_classical=args.classical
    )

    print(f"Detected {len(lines)} line(s) via {method}")
    base = os.path.splitext(os.path.basename(args.image))[0]
    for i, line_img in enumerate(lines, 1):
        out_path = os.path.join(args.out_dir, f"{base}_line{i:02d}.jpg")
        cv2.imwrite(out_path, line_img)
        print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
