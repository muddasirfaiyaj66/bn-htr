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

from dataset import IMG_MAX_WIDTH_INFER, preprocess_image
from decode import decode_log_probs, load_lm, logits_to_log_probs
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
    ap.add_argument("--lm", default=None, help="Character LM json. Defaults to data/lm.json when present")
    ap.add_argument("--greedy", action="store_true", help="Use greedy CTC instead of beam search")
    ap.add_argument("--enhanced", action="store_true", help="Use preprocess.preprocess_line (colour ink, CLAHE, crop)")
    ap.add_argument("--sauvola", action="store_true", help="Sauvola-binarize inside the enhanced preprocessor")
    ap.add_argument("--channel", default="auto", help="Ink channel: auto, gray, min, green, lab_l")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)

    idx2char = ckpt["idx2char"]
    num_classes = len(ckpt["char2idx"]) + 1

    model = CRNN(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    lm = None if args.greedy else load_lm(args.lm)

    clean = bool(ckpt.get("clean", False))
    binarize = bool(ckpt.get("binarize", False))

    def recognize(img_path):
        img = preprocess_image(
            img_path,
            augment=False,
            clean=clean,
            binarize=binarize,
            enhanced=args.enhanced,
            sauvola=args.sauvola,
            channel=args.channel,
            allow_wide=True,
            max_w=IMG_MAX_WIDTH_INFER,
        )
        tensor = torch.from_numpy(img).unsqueeze(0).to(device)  # (1, 1, H, W)
        with torch.no_grad():
            logits = model(tensor)[0]  # (T, C)
        if lm is None:
            return ctc_greedy_decode_single(logits, idx2char)
        log_probs = logits_to_log_probs(logits.detach().float().cpu().numpy())
        return decode_log_probs(log_probs, idx2char, lm=lm)

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
