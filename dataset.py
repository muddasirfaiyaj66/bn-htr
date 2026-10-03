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


def preprocess_array(img, target_h=IMG_HEIGHT, max_w=None, augment=False, clean=False, binarize=False):
    """Convert a grayscale uint8 image to a (1, H, W) float32 tensor."""
    if max_w is None:
        max_w = IMG_MAX_WIDTH
    if img is None or img.size == 0:
        raise ValueError("Empty image passed to preprocess_array")
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

    padded = padded.astype(np.float32) / 255.0
    padded = (padded - 0.5) / 0.5
    return padded[np.newaxis, :, :]


def preprocess_image(img_path, target_h=IMG_HEIGHT, max_w=None, augment=False, clean=False, binarize=False):
    if max_w is None:
        max_w = IMG_MAX_WIDTH
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    return preprocess_array(
        img, target_h=target_h, max_w=max_w, augment=augment, clean=clean, binarize=binarize
    )


def _augment(img):
    h, w = img.shape

    if random.random() < 0.6:
        angle = random.uniform(-3.0, 3.0)
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, matrix, (w, h), borderValue=255)

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

    if random.random() < 0.4:
        noise = np.random.normal(0, random.uniform(4, 12), img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img


class BNHTRDataset(Dataset):
    def __init__(self, csv_path, char2idx, augment=False, clean=False, binarize=False):
        self.rows = load_manifest(csv_path)
        self.char2idx = char2idx
        self.augment = augment
        self.clean = clean
        self.binarize = binarize

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        img_path, text = self.rows[idx]
        img = preprocess_image(
            img_path, augment=self.augment, clean=self.clean, binarize=self.binarize
        )
        label = [self.char2idx[c] for c in text if c in self.char2idx]
        return torch.from_numpy(img), torch.tensor(label, dtype=torch.long), text


def collate_fn(batch):
    imgs, labels, texts = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels_concat = torch.cat(labels)
    return imgs, labels_concat, label_lengths, texts
