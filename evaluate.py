"""
evaluate.py

Runs the trained model on the held-out TEST set (writers never seen during
training or validation) and reports final CER / WER, plus a handful of
example predictions vs ground truth so you can eyeball quality.

Also scores a folder of real line photos, each paired with a UTF-8 .txt label
of the same stem.

Usage:
    python evaluate.py --checkpoint checkpoints\\best.pt --test_csv data\\test.csv
    python evaluate.py --checkpoint checkpoints\\best.pt --samples_dir tests\\real_samples --out_json results\\baseline.json
"""

import argparse
import json
import os
import random

import editdistance
import torch
from torch.utils.data import DataLoader

from dataset import BNHTRDataset, collate_fn
from decode import decode_log_probs, load_lm, logits_to_log_probs
from model import CRNN
from train import ctc_greedy_decode  # reuse the same decoder used during training

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")


def score_texts(predictions, targets):
    """Aggregate character and word error rates for paired strings."""
    total_cer_dist, total_cer_len = 0, 0
    total_wer_dist, total_wer_len = 0, 0
    for pred, target in zip(predictions, targets):
        pred = pred or ""
        target = target or ""
        total_cer_dist += editdistance.eval(pred, target)
        total_cer_len += max(len(target), 1)
        pred_words = pred.split()
        target_words = target.split()
        total_wer_dist += editdistance.eval(pred_words, target_words)
        total_wer_len += max(len(target_words), 1)
    n = len(list(targets))
    return {
        "n": n,
        "cer": total_cer_dist / max(total_cer_len, 1),
        "wer": total_wer_dist / max(total_wer_len, 1),
        "cer_dist": total_cer_dist,
        "cer_len": total_cer_len,
        "wer_dist": total_wer_dist,
        "wer_len": total_wer_len,
    }


def load_sample_pairs(folder):
    """Return (image_path, ground_truth) for every image that has a sibling .txt."""
    folder = os.path.abspath(folder)
    pairs = []
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Sample folder not found: {folder}")
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not name.lower().endswith(_IMAGE_EXTS) or not os.path.isfile(path):
            continue
        stem, _ext = os.path.splitext(path)
        label_path = stem + ".txt"
        if not os.path.isfile(label_path):
            raise FileNotFoundError(f"Missing label for {name}: {label_path}")
        with open(label_path, "r", encoding="utf-8") as handle:
            text = handle.read().strip()
        pairs.append((path, text))
    if not pairs:
        raise FileNotFoundError(f"No labeled images in {folder}")
    return pairs


def evaluate_sample_dir(recognize_fn, folder):
    """Score recognize_fn(image_path) -> str against sibling .txt labels."""
    rows = []
    preds, targets = [], []
    for path, target in load_sample_pairs(folder):
        pred = recognize_fn(path) or ""
        preds.append(pred)
        targets.append(target)
        rows.append(
            {
                "image": os.path.basename(path),
                "target": target,
                "prediction": pred,
            }
        )
    metrics = score_texts(preds, targets)
    metrics["samples"] = rows
    return metrics


def write_metrics(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test_csv", default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--num_examples", type=int, default=10,
                     help="How many sample predictions to print")
    ap.add_argument("--beam", action="store_true", help="Decode with the character LM beam search")
    ap.add_argument("--lm", default=None)
    ap.add_argument("--lexicon", action="store_true")
    ap.add_argument(
        "--samples_dir",
        default=None,
        help="Folder of line images with a same-stem UTF-8 .txt ground truth",
    )
    ap.add_argument("--out_json", default=None, help="Write sample-folder metrics to this JSON file")
    ap.add_argument("--enhanced", action="store_true", help="Score samples with preprocess_line")
    ap.add_argument("--sauvola", action="store_true", help="Sauvola threshold inside preprocess_line")
    ap.add_argument("--channel", default="auto")
    args = ap.parse_args()

    if not args.test_csv and not args.samples_dir:
        ap.error("Provide --test_csv and/or --samples_dir")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.test_csv:
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

    if args.samples_dir:
        from recognize import build_crnn_recognizer, load_torch_checkpoint

        rec_ckpt = load_torch_checkpoint(args.checkpoint, device)
        recognizer = build_crnn_recognizer(
            rec_ckpt, device, enhanced=args.enhanced, sauvola=args.sauvola, channel=args.channel
        )
        metrics = evaluate_sample_dir(recognizer.recognize_line_path, args.samples_dir)
        print(f"\n=== Real samples ({metrics['n']} lines) ===")
        print(f"Character Error Rate (CER): {metrics['cer']:.4f}  ({metrics['cer']*100:.2f}%)")
        print(f"Word Error Rate (WER):      {metrics['wer']:.4f}  ({metrics['wer']*100:.2f}%)")
        for row in metrics["samples"]:
            print(f"  Ground truth: {row['target']}")
            print(f"  Predicted:    {row['prediction']}")
        payload = {
            "checkpoint": os.path.abspath(args.checkpoint),
            "samples_dir": os.path.abspath(args.samples_dir),
            "preprocess": "enhanced+sauvola" if args.enhanced and args.sauvola else ("enhanced" if args.enhanced else "legacy"),
            "channel": args.channel,
            "cer": metrics["cer"],
            "wer": metrics["wer"],
            "n": metrics["n"],
            "samples": metrics["samples"],
        }
        if args.out_json:
            write_metrics(args.out_json, payload)
            print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
