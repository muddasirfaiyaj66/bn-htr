"""
evaluate.py

Runs the trained model on the held-out TEST set (writers never seen during
training or validation) and reports final CER / WER, plus a handful of
example predictions vs ground truth so you can eyeball quality.

Usage:
    python evaluate.py --checkpoint checkpoints\\best.pt --test_csv data\\test.csv
"""

import argparse
import random

import editdistance
import torch
from torch.utils.data import DataLoader

from dataset import BNHTRDataset, collate_fn
from decode import decode_log_probs, load_lm, logits_to_log_probs
from model import CRNN
from train import ctc_greedy_decode  # reuse the same decoder used during training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--num_examples", type=int, default=10,
                     help="How many sample predictions to print")
    ap.add_argument("--beam", action="store_true", help="Decode with the character LM beam search")
    ap.add_argument("--lm", default=None)
    ap.add_argument("--lexicon", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)

    char2idx = ckpt["char2idx"]
    idx2char = ckpt["idx2char"]
    num_classes = len(char2idx) + 1

    model = CRNN(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    lm = load_lm(args.lm) if args.beam else None
    if args.beam and lm is None:
        raise FileNotFoundError("Beam decoding needs data/lm.json. Build it with decode.py.")

    test_ds = BNHTRDataset(args.test_csv, char2idx, augment=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn)

    total_cer_dist, total_cer_len = 0, 0
    total_wer_dist, total_wer_len = 0, 0
    all_examples = []

    with torch.no_grad():
        for imgs, labels_concat, label_lengths, texts in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            if lm is None:
                preds = ctc_greedy_decode(logits, idx2char)
            else:
                log_probs = logits_to_log_probs(logits.detach().float().cpu().numpy())
                preds = [
                    decode_log_probs(
                        log_probs[i], idx2char, lm=lm, lexicon=args.lexicon
                    )
                    for i in range(log_probs.shape[0])
                ]

            for pred, target in zip(preds, texts):
                total_cer_dist += editdistance.eval(pred, target)
                total_cer_len += max(len(target), 1)

                pred_words = pred.split()
                target_words = target.split()
                total_wer_dist += editdistance.eval(pred_words, target_words)
                total_wer_len += max(len(target_words), 1)

                all_examples.append((target, pred))

    cer = total_cer_dist / total_cer_len
    wer = total_wer_dist / total_wer_len

    print(f"\n=== Test set results ({len(test_ds)} lines) ===")
    print(f"Character Error Rate (CER): {cer:.4f}  ({cer*100:.2f}%)")
    print(f"Word Error Rate (WER):      {wer:.4f}  ({wer*100:.2f}%)")

    print(f"\n=== {min(args.num_examples, len(all_examples))} random example predictions ===")
    for target, pred in random.sample(all_examples, min(args.num_examples, len(all_examples))):
        print(f"  Ground truth: {target}")
        print(f"  Predicted:    {pred}")
        print()


if __name__ == "__main__":
    main()
