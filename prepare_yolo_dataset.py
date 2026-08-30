"""
prepare_yolo_dataset.py

Build an Ultralytics YOLO dataset from BN-HTRd page images and line boxes.

Usage:
    python prepare_yolo_dataset.py --dataset_root path\\to\\BN-HTR_Dataset --out_dir data\\yolo_lines
"""

import argparse
import os
import random
import shutil
from pathlib import Path


def find_page_pairs(lines_root: Path):
    """Yield (page.jpg, page.txt) pairs that both exist."""
    pairs = []
    for doc_dir in sorted(lines_root.iterdir()):
        if not doc_dir.is_dir():
            continue
        for jpg in sorted(doc_dir.glob("*.jpg")):
            txt = jpg.with_suffix(".txt")
            if txt.is_file():
                pairs.append((jpg, txt))
    return pairs


def remap_label(src: Path, dst: Path):
    """Rewrite labels with class id 0 (single class: line)."""
    lines_out = []
    with open(src, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split()
            if len(parts) != 5:
                continue
            _, cx, cy, w, h = parts
            lines_out.append(f"0 {cx} {cy} {w} {h}\n")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        f.writelines(lines_out)


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset_root",
        required=True,
        help="Path to BN-HTR_Dataset (contains Segmentation_Images/)",
    )
    ap.add_argument("--out_dir", default="data/yolo_lines")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    lines_root = Path(args.dataset_root) / "Segmentation_Images" / "Lines"
    if not lines_root.is_dir():
        raise FileNotFoundError(f"Missing Lines folder: {lines_root}")

    pairs = find_page_pairs(lines_root)
    if not pairs:
        raise RuntimeError(f"No labeled page pairs found under {lines_root}")

    random.Random(args.seed).shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_ratio))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    out = Path(args.out_dir)
    if out.exists():
        shutil.rmtree(out)

    for split, subset in (("train", train_pairs), ("val", val_pairs)):
        for jpg, txt in subset:
            stem = f"{jpg.parent.name}_{jpg.stem}"
            link_or_copy(jpg, out / "images" / split / f"{stem}.jpg")
            remap_label(txt, out / "labels" / split / f"{stem}.txt")

    yaml_path = out / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: line",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Prepared YOLO dataset at {out.resolve()}")
    print(f"  train pages: {len(train_pairs)}")
    print(f"  val pages:   {len(val_pairs)}")
    print(f"  data.yaml:   {yaml_path}")


if __name__ == "__main__":
    main()
