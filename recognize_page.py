"""
recognize_page.py

End-to-end CLI: handwritten image → Bangla text.

Usage:
    python recognize_page.py --checkpoint checkpoints\\best.pt --image path\\to\\page.jpg
"""

import argparse
import os

import cv2
import torch

from model import CRNN
from recognize import Recognizer
from segment_lines import segment_page


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--save_lines_dir", default=None, help="Save cropped line images")
    ap.add_argument("--force_page", action="store_true")
    ap.add_argument("--force_line", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location=device)
    idx2char = ckpt["idx2char"]
    num_classes = len(ckpt["char2idx"]) + 1

    model = CRNN(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    recognizer = Recognizer(model, idx2char, device)

    force = None
    if args.force_page:
        force = "page"
    elif args.force_line:
        force = "line"

    if args.save_lines_dir:
        os.makedirs(args.save_lines_dir, exist_ok=True)
        _, line_imgs, method = segment_page(args.image)
        print(f"Segmenter={method}, lines={len(line_imgs)}")
        for i, line_img in enumerate(line_imgs, 1):
            cv2.imwrite(
                os.path.join(args.save_lines_dir, f"line_{i:02d}.jpg"), line_img
            )

    result = recognizer.recognize(args.image, force_mode=force)
    print(
        f"mode={result['mode']} segmenter={result['segmenter']} "
        f"lines={result['line_count']}\n"
    )
    for i, text in enumerate(result["lines"], 1):
        print(f"Line {i}: {text}")
    print("\n--- Full recognized text ---")
    print(result["full_text"])


if __name__ == "__main__":
    main()
