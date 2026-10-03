"""
synth_lines.py

Render extra Bangla line images from a text corpus and handwriting-style fonts.
The same training augmentations used by dataset.py are applied afterwards.

PIL draws the glyphs. Conjunct shaping is correct only when Pillow was built
with libraqm; otherwise the image is still a useful ink target for the string.

Usage:
    python synth_lines.py --corpus data\\train.csv --out_dir data\\synth --count 100
"""

from __future__ import annotations

import argparse
import csv
import os
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dataset import _augment

_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\Nirmala.ttc",
    r"C:\Windows\Fonts\kalpurush.ttf",
    r"C:\Windows\Fonts\Siyamrupali.ttf",
    r"C:\Windows\Fonts\vrinda.ttf",
    "/usr/share/fonts/truetype/lohit-bengali/Lohit-Bengali.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
)


def find_fonts() -> list[str]:
    """Return installed font files that can render Bangla, when any are present."""
    found = [path for path in _FONT_CANDIDATES if os.path.isfile(path)]
    fonts_dir = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    if os.path.isdir(fonts_dir):
        for name in os.listdir(fonts_dir):
            lower = name.lower()
            if any(key in lower for key in ("nirmala", "kalpurush", "siyam", "vrinda", "bangla")):
                path = os.path.join(fonts_dir, name)
                if path not in found:
                    found.append(path)
    return found


def load_sentences(path: str, limit: int) -> list[str]:
    rows = []
    if path.lower().endswith(".csv"):
        with open(path, "r", encoding="utf-8") as handle:
            for record in csv.DictReader(handle):
                text = " ".join((record.get("text") or "").split())
                if text:
                    rows.append(text)
    else:
        with open(path, "r", encoding="utf-8") as handle:
            rows = [" ".join(line.split()) for line in handle if line.strip()]
    if not rows:
        raise FileNotFoundError(f"No text in {path}")
    if len(rows) > limit:
        return random.sample(rows, limit)
    return rows


def render_line(text: str, font_path: str, size: int = 42) -> np.ndarray:
    """Draw `text` in dark ink on a white line. Returns uint8 grayscale."""
    font = ImageFont.truetype(font_path, size=size)
    probe = Image.new("L", (8, 8), 255)
    box = ImageDraw.Draw(probe).textbbox((0, 0), text, font=font)
    width = max((box[2] - box[0]) + 36, 48)
    height = max((box[3] - box[1]) + 24, 48)
    image = Image.new("L", (width, height), 255)
    ImageDraw.Draw(image).text((16 - box[0], 12 - box[1]), text, font=font, fill=20)
    return np.asarray(image, dtype=np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render synthetic Bangla handwriting lines")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out_dir", default="data/synth")
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--font", action="append", default=None, help="Font file. Repeat to use several.")
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    fonts = args.font or find_fonts()
    fonts = [path for path in fonts if os.path.isfile(path)]
    if not fonts:
        raise SystemExit("No Bangla font found. Pass --font path\\to\\kalpurush.ttf")

    os.makedirs(args.out_dir, exist_ok=True)
    sentences = load_sentences(args.corpus, args.count)
    manifest = os.path.join(args.out_dir, "synth.csv")
    with open(manifest, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "text"])
        writer.writeheader()
        for index, text in enumerate(sentences):
            gray = render_line(text, random.choice(fonts), size=random.randint(36, 52))
            gray = _augment(gray)
            name = f"synth_{index:05d}.png"
            path = os.path.join(args.out_dir, name)
            cv2.imwrite(path, gray)
            writer.writerow({"image_path": os.path.abspath(path), "text": text})
    print(f"Wrote {len(sentences)} lines to {manifest}")


if __name__ == "__main__":
    main()
