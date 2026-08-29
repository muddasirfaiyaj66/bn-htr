"""
dataset.py

PyTorch Dataset for BN-HTRd line images + a collate_fn suited for CTC training
(variable-length label sequences, fixed-size padded images).
"""

import csv
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_HEIGHT = 64
IMG_MAX_WIDTH = 800


def load_manifest(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["image_path"], row["text"]))
    return rows


def preprocess_image(img_path, target_h=IMG_HEIGHT, max_w=IMG_MAX_WIDTH, augment=False):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    if augment:
        img = _augment(img)

    h, w = img.shape
    scale = target_h / h
    new_w = min(max(1, int(w * scale)), max_w)
    img = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

    padded = np.ones((target_h, max_w), dtype=np.uint8) * 255  # white background
    padded[:, :new_w] = img

    padded = padded.astype(np.float32) / 255.0
    padded = (padded - 0.5) / 0.5  # normalize to [-1, 1]
    return padded[np.newaxis, :, :]  # (1, H, W) channel-first


def _augment(img):
    # mild random rotation
    if random.random() < 0.5:
        angle = random.uniform(-2.5, 2.5)
        h, w = img.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=255)

    # random erosion/dilation to simulate pen thickness variation
    if random.random() < 0.3:
        kernel = np.ones((2, 2), np.uint8)
        if random.random() < 0.5:
            img = cv2.erode(img, kernel, iterations=1)
        else:
            img = cv2.dilate(img, kernel, iterations=1)

    # random brightness/contrast jitter
    if random.random() < 0.5:
        alpha = random.uniform(0.85, 1.15)  # contrast
        beta = random.uniform(-15, 15)      # brightness
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
    imgs = torch.stack(imgs, dim=0)  # (B, 1, H, W) — all already padded to same width
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels_concat = torch.cat(labels)  # CTCLoss wants a flat concatenated tensor
    return imgs, labels_concat, label_lengths, texts
