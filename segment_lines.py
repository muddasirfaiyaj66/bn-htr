"""
segment_lines.py

Automatically detects how many text lines are on a handwritten page
(no manual line-count input). Estimates line spacing from the page's
horizontal ink profile, finds one peak per line, then crops each band.

Fallback: connected-components + OPTICS if the profile is too weak.

Usage:
    python segment_lines.py --image path\\to\\full_page.jpg --out_dir segmented_lines
"""

import argparse
import os

import cv2
import numpy as np
from sklearn.cluster import OPTICS


def _binarize(img):
    """Return ink=255, background=0 binary image."""
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.dilate(opened, kernel, iterations=1)


def _estimate_line_pitch(smooth):
    """
    Automatically estimate average vertical distance between lines from the
    autocorrelation of the ink projection. Returns pitch in pixels, or None.
    """
    n = len(smooth)
    if n < 40 or smooth.max() <= 0:
        return None

    x = smooth - float(smooth.mean())
    # FFT autocorrelation
    fft = np.fft.rfft(x, n=2 * n)
    ac = np.fft.irfft(fft * np.conj(fft))[:n]
    if ac[0] <= 1e-8:
        return None
    ac = ac / ac[0]

    # Plausible line pitch range for a page scan
    lo = max(18, n // 100)
    hi = max(lo + 5, min(n // 3, 220))
    region = ac[lo:hi]
    if region.size == 0:
        return None

    peak_rel = int(np.argmax(region))
    pitch = lo + peak_rel
    # Require a real periodic bump; otherwise pitch guess is unreliable
    if ac[pitch] < 0.12:
        return None
    return int(pitch)


def _find_peaks(smooth, min_distance, prominence_ratio=0.08):
    """Pick local maxima that are tall enough and spaced by ~one line."""
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
        valley = a + int(np.argmin(smooth[a:b + 1]))
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
    """Crop left/right whitespace so page lines look closer to training crops."""
    if crop.size == 0:
        return crop
    # ink = dark pixels
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
        # Keep short/centered titles: require a little ink, not full-page density
        min_ink = max(60, crop.shape[0] * 2)
        if ink < min_ink:
            continue
        crops.append(crop)
    return crops


def _segment_by_projection(img, binary):
    """
    Auto-detect line count from ink projection:
      1) estimate line pitch (spacing) from the page itself
      2) find one peak per line using that pitch
      3) cut valleys between peaks into line bands
    """
    page_h = img.shape[0]
    row_ink = (binary > 0).sum(axis=1).astype(np.float32)

    k = max(5, (int(page_h * 0.006) | 1))
    smooth = np.convolve(row_ink, np.ones(k, dtype=np.float32) / k, mode="same")

    pitch = _estimate_line_pitch(smooth)
    if pitch is None:
        # Weak periodicity — still search peaks with a page-relative gap
        min_distance = max(10, page_h // 50)
        merge_sep = max(12, page_h // 70)
    else:
        # Peaks must be ~one line apart; merge anything closer than ~half pitch
        min_distance = max(8, int(0.55 * pitch))
        merge_sep = max(10, int(0.40 * pitch))

    peaks = _find_peaks(smooth, min_distance=min_distance, prominence_ratio=0.07)
    peaks = _merge_close_peaks(peaks, smooth, min_sep=merge_sep)

    # If pitch is known but we found far too few/many peaks, re-pick once
    # with a slightly adjusted distance (still fully automatic).
    if pitch is not None and peaks:
        expected = max(1, int(round(page_h / pitch)))
        if len(peaks) > expected * 1.6:
            peaks = _find_peaks(smooth, min_distance=max(8, int(0.7 * pitch)), prominence_ratio=0.09)
            peaks = _merge_close_peaks(peaks, smooth, min_sep=max(10, int(0.45 * pitch)))
        elif len(peaks) < max(2, expected * 0.45):
            peaks = _find_peaks(smooth, min_distance=max(6, int(0.4 * pitch)), prominence_ratio=0.05)
            peaks = _merge_close_peaks(peaks, smooth, min_sep=max(8, int(0.35 * pitch)))

    return _peak_bands(smooth, peaks, page_h)


def _segment_by_optics(img, binary, min_area=80):
    """Fallback auto clustering when projection fails."""
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


def segment_page(img_path, min_area=80):
    """
    Automatically detect and crop every text line on a page.
    Line count is inferred from the image — never passed in manually.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(img_path)

    target_w = 1200
    scale = target_w / img.shape[1]
    img = cv2.resize(img, (target_w, int(img.shape[0] * scale)))

    binary = _binarize(img)
    bands = _segment_by_projection(img, binary)

    if len(bands) < 1:
        bands = _segment_by_optics(img, binary, min_area=min_area)

    line_images = _bands_to_crops(img, bands)
    return img, line_images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    _, lines = segment_page(args.image)

    print(f"Automatically detected {len(lines)} line(s)")
    base = os.path.splitext(os.path.basename(args.image))[0]
    for i, line_img in enumerate(lines, 1):
        out_path = os.path.join(args.out_dir, f"{base}_line{i:02d}.jpg")
        cv2.imwrite(out_path, line_img)
        print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
