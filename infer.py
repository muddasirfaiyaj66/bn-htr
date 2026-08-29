"""
infer.py

Run the trained CRNN on one or more line images and print recognized text.

Usage:
    python infer.py --checkpoint checkpoints\\best.pt --image path\\to\\line.jpg
    python infer.py --checkpoint checkpoints\\best.pt --dir path\\to\\folder_of_lines
"""

import argparse
import os

import torch

from dataset import preprocess_image
from model import CRNN


def ctc_greedy_decode_single(logits, idx2char):
    preds = logits.argmax(dim=1).tolist()  # (T,)
    prev = -1
    chars = []
    for idx in preds:
        if idx != prev and idx != 0:
            chars.append(idx2char.get(idx, ""))
        prev = idx
    return "".join(chars)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", help="Path to a single line image")
    ap.add_argument("--dir", help="Path to a folder of line images (processed in filename order)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)

    idx2char = ckpt["idx2char"]
    num_classes = len(ckpt["char2idx"]) + 1

    model = CRNN(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    def recognize(img_path):
        img = preprocess_image(img_path, augment=False)
        tensor = torch.from_numpy(img).unsqueeze(0).to(device)  # (1, 1, H, W)
        with torch.no_grad():
            logits = model(tensor)[0]  # (T, C)
        return ctc_greedy_decode_single(logits, idx2char)

    if args.image:
        print(recognize(args.image))
    elif args.dir:
        files = sorted(f for f in os.listdir(args.dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        for fname in files:
            text = recognize(os.path.join(args.dir, fname))
            print(f"{fname}: {text}")
    else:
        print("Provide --image or --dir")


if __name__ == "__main__":
    main()
