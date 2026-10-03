"""
dataset.py

PyTorch Dataset and image preprocessing for CRNN+CTC training.
"""

import csv
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from normalize import normalize_bangla

IMG_HEIGHT = 64
IMG_MAX_WIDTH = 800
IMG_MAX_WIDTH_INFER = 1280


def load_manifest(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["image_path"], row["text"]))
    return rows


def flatten_illumination(img):
    """Whiten uneven paper lighting while keeping stroke gray levels."""
    bg = cv2.GaussianBlur(img, (0, 0), sigmaX=21, sigmaY=21)
    bg = np.maximum(bg, 1)
    flat = cv2.divide(img, bg, scale=255)
    return np.clip(flat, 0, 255).astype(np.uint8)


def deskew_line(img, max_angle=12.0):
    """Straighten a line crop using the ink orientation. Skips near-level lines."""
    ink = np.column_stack(np.where(img < 190))
    if ink.shape[0] < 40:
        return img
    coords = np.column_stack([ink[:, 1], ink[:, 0]]).astype(np.float32)
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.4 or abs(angle) > max_angle:
        return img
    h, w = img.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)


def clean_line(img, binarize=False):
    """Deskew and flatten illumination. Hard binarization is optional."""
    img = deskew_line(img)
    img = flatten_illumination(img)
    if binarize:
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img


def _to_model_tensor(img):
    """Map a uint8 line image to the CRNN's float range."""
    padded = img.astype(np.float32) / 255.0
    padded = (padded - 0.5) / 0.5
    return padded[np.newaxis, :, :]


def preprocess_array(
    img,
    target_h=IMG_HEIGHT,
    max_w=None,
    augment=False,
    clean=False,
    binarize=False,
    enhanced=False,
    sauvola=False,
    channel="auto",
    allow_wide=False,
):
    """Convert a line image to a (1, H, W) float32 tensor."""
    if max_w is None:
        max_w = IMG_MAX_WIDTH
    if img is None or img.size == 0:
        raise ValueError("Empty image passed to preprocess_array")

    if enhanced:
        from preprocess import PreprocessConfig, preprocess_line

        if augment:
            img = _augment(img)
        prepared = preprocess_line(
            img,
            PreprocessConfig(
                target_h=target_h,
                max_w=max_w,
                sauvola=sauvola,
                channel=channel,
                allow_wide=allow_wide,
            ),
        )
        return _to_model_tensor(prepared)

    if img.ndim == 3:
        if img.shape[2] == 1:
            img = img[:, :, 0]
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim != 2:
        raise ValueError(f"Expected HxW or HxWxC image, got shape {img.shape}")

    if clean or binarize:
        img = clean_line(img, binarize=binarize)

    if augment:
        img = _augment(img)

    h, w = img.shape
    scale = target_h / h
    new_w = min(max(1, int(w * scale)), max_w)
    img = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

    padded = np.ones((target_h, max_w), dtype=np.uint8) * 255
    padded[:, :new_w] = img
    return _to_model_tensor(padded)


def preprocess_image(
    img_path,
    target_h=IMG_HEIGHT,
    max_w=None,
    augment=False,
    clean=False,
    binarize=False,
    enhanced=False,
    sauvola=False,
    channel="auto",
    allow_wide=False,
):
    if max_w is None:
        max_w = IMG_MAX_WIDTH
    flag = cv2.IMREAD_COLOR if enhanced else cv2.IMREAD_GRAYSCALE
    img = cv2.imread(img_path, flag)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    return preprocess_array(
        img,
        target_h=target_h,
        max_w=max_w,
        augment=augment,
        clean=clean,
        binarize=binarize,
        enhanced=enhanced,
        sauvola=sauvola,
        channel=channel,
        allow_wide=allow_wide,
    )


def _to_gray(img):
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _augment_color(img):
    """Recolour dark strokes as black, blue, or red ink on textured paper."""
    gray = _to_gray(img)
    height, width = gray.shape
    paper = np.full((height, width, 3), random.randint(200, 245), np.uint8)
    texture = np.random.normal(0, random.uniform(2, 8), paper.shape)
    paper = np.clip(paper.astype(np.float32) + texture, 0, 255).astype(np.uint8)
    ink = random.choice(((20, 20, 20), (140, 40, 20), (30, 30, 160)))
    paper[gray < 170] = ink
    top, bottom = random.randint(0, 8), random.randint(0, 8)
    left, right = random.randint(0, 12), random.randint(0, 12)
    canvas = np.full((height + top + bottom, width + left + right, 3), 235, np.uint8)
    canvas[top : top + height, left : left + width] = paper
    return canvas


def _jpeg_roundtrip(img):
    quality = random.randint(35, 90)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img
    flag = cv2.IMREAD_GRAYSCALE if img.ndim == 2 else cv2.IMREAD_COLOR
    decoded = cv2.imdecode(buf, flag)
    return img if decoded is None else decoded


def _augment(img):
    """Training-only geometry, blur, noise, JPEG, and coloured-ink simulation."""
    if random.random() < 0.45:
        img = _augment_color(img)
    img = _to_gray(img)
    h, w = img.shape

    if random.random() < 0.6:
        angle = random.uniform(-3.0, 3.0)
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, matrix, (w, h), borderValue=255)

    if random.random() < 0.35:
        src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        jx = lambda: random.uniform(-0.05, 0.05) * w
        jy = lambda: random.uniform(-0.08, 0.08) * h
        dst = np.float32(
            [[jx(), jy()], [w - 1 + jx(), jy()], [w - 1 + jx(), h - 1 + jy()], [jx(), h - 1 + jy()]]
        )
        matrix = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, matrix, (w, h), borderValue=255)

    if random.random() < 0.5:
        shear = random.uniform(-0.3, 0.3)
        matrix = np.array([[1, shear, 0], [0, 1, 0]], dtype=np.float32)
        img = cv2.warpAffine(img, matrix, (w, h), borderValue=255)

    if random.random() < 0.3:
        alpha = random.uniform(2.0, 6.0)
        sigma = random.uniform(2.0, 4.0)
        dx = cv2.GaussianBlur((np.random.rand(h, w).astype(np.float32) * 2 - 1), (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur((np.random.rand(h, w).astype(np.float32) * 2 - 1), (0, 0), sigma) * alpha
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        img = cv2.remap(img, xs + dx, ys + dy, cv2.INTER_LINEAR, borderValue=255)

    if random.random() < 0.3:
        kernel = np.ones((2, 2), np.uint8)
        if random.random() < 0.5:
            img = cv2.erode(img, kernel, iterations=1)
        else:
            img = cv2.dilate(img, kernel, iterations=1)

    if random.random() < 0.5:
        alpha = random.uniform(0.8, 1.2)
        beta = random.uniform(-18, 18)
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    if random.random() < 0.3:
        kernel = random.choice((3, 5))
        img = cv2.GaussianBlur(img, (kernel, kernel), 0)

    if random.random() < 0.25:
        length = random.choice((5, 7, 9))
        kernel = np.zeros((length, length), np.float32)
        kernel[length // 2, :] = 1.0 / length
        img = cv2.filter2D(img, -1, kernel)

    if random.random() < 0.4:
        noise = np.random.normal(0, random.uniform(4, 12), img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if random.random() < 0.35:
        img = _jpeg_roundtrip(img)

    if random.random() < 0.3:
        top, bottom = random.randint(0, 6), random.randint(0, 6)
        left, right = random.randint(0, 10), random.randint(0, 10)
        canvas = np.full((h + top + bottom, w + left + right), 255, np.uint8)
        canvas[top : top + img.shape[0], left : left + img.shape[1]] = img
        img = canvas

    return img


class BNHTRDataset(Dataset):
    def __init__(self, csv_path, char2idx, augment=False, clean=False, binarize=False, enhanced=False, sauvola=False, channel="auto"):
        self.rows = load_manifest(csv_path)
        self.char2idx = char2idx
        self.augment = augment
        self.clean = clean
        self.binarize = binarize
        self.enhanced = enhanced
        self.sauvola = sauvola
        self.channel = channel

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        img_path, text = self.rows[idx]
        text = normalize_bangla(text)
        img = preprocess_image(
            img_path,
            augment=self.augment,
            clean=self.clean,
            binarize=self.binarize,
            enhanced=self.enhanced,
            sauvola=self.sauvola,
            channel=self.channel,
            allow_wide=False,
        )
        label = [self.char2idx[c] for c in text if c in self.char2idx]
        return torch.from_numpy(img), torch.tensor(label, dtype=torch.long), text


def collate_fn(batch):
    imgs, labels, texts = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels_concat = torch.cat(labels)
    return imgs, labels_concat, label_lengths, texts
