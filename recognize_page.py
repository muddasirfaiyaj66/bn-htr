"""
recognize_page.py

End-to-end: full handwritten page image -> segmented lines -> recognized Bangla text.

Usage:
    python recognize_page.py --checkpoint checkpoints\\best.pt --image path\\to\\page.jpg
"""

import argparse
import tempfile
import os

import cv2
import torch

from dataset import preprocess_image
from model import CRNN
from segment_lines import segment_page
from infer import ctc_greedy_decode_single


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--save_lines_dir", default=None, help="Optional: save cropped line images here")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    idx2char = ckpt["idx2char"]
    num_classes = len(ckpt["char2idx"]) + 1

    model = CRNN(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    _, line_imgs = segment_page(args.image)
    print(f"Detected {len(line_imgs)} lines\n")

    save_dir = args.save_lines_dir
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    full_text_lines = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, line_img in enumerate(line_imgs, 1):
            tmp_path = os.path.join(tmp, f"line_{i}.jpg")
            cv2.imwrite(tmp_path, line_img)

            if save_dir:
                cv2.imwrite(os.path.join(save_dir, f"line_{i:02d}.jpg"), line_img)

            img = preprocess_image(tmp_path, augment=False)
            tensor = torch.from_numpy(img).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor)[0]
            text = ctc_greedy_decode_single(logits, idx2char)
            full_text_lines.append(text)
            print(f"Line {i}: {text}")

    print("\n--- Full recognized text ---")
    print("\n".join(full_text_lines))


if __name__ == "__main__":
    main()
