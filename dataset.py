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


def preprocess_array(img, target_h=IMG_HEIGHT, max_w=None, augment=False):
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


def preprocess_image(img_path, target_h=IMG_HEIGHT, max_w=None, augment=False):
    if max_w is None:
        max_w = IMG_MAX_WIDTH
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    return preprocess_array(img, target_h=target_h, max_w=max_w, augment=augment)


def _augment(img):
    if random.random() < 0.5:
        angle = random.uniform(-2.5, 2.5)
        h, w = img.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=255)

    if random.random() < 0.3:
        kernel = np.ones((2, 2), np.uint8)
        if random.random() < 0.5:
            img = cv2.erode(img, kernel, iterations=1)
        else:
            img = cv2.dilate(img, kernel, iterations=1)

    if random.random() < 0.5:
        alpha = random.uniform(0.85, 1.15)
        beta = random.uniform(-15, 15)
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    return img


class BNHTRDataset(Dataset):
    def __init__(self, csv_path, char2idx, augment=False):
        self.rows = load_manifest(csv_path)
        self.char2idx = char2idx
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        img_path, text = self.rows[idx]
        img = preprocess_image(img_path, augment=self.augment)
        label = [self.char2idx[c] for c in text if c in self.char2idx]
        return torch.from_numpy(img), torch.tensor(label, dtype=torch.long), text


def collate_fn(batch):
    imgs, labels, texts = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels_concat = torch.cat(labels)
    return imgs, labels_concat, label_lengths, texts
