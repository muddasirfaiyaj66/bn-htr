"""
vocab.py

Builds and stores the character vocabulary used for CTC training.
Index 0 is reserved for the CTC blank token.
"""

import csv
import json


def build_vocab_from_csv(csv_paths, out_json_path):
    chars = set()
    for path in csv_paths:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                chars.update(row["text"])

    vocab = sorted(chars)
    char2idx = {c: i + 1 for i, c in enumerate(vocab)}  # 0 = blank
    idx2char = {i + 1: c for i, c in enumerate(vocab)}

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump({"char2idx": char2idx, "idx2char": idx2char}, f, ensure_ascii=False, indent=2)

    print(f"Vocabulary size (excluding blank): {len(vocab)}")
    print(f"Saved to {out_json_path}")
    return char2idx, idx2char


def load_vocab(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    char2idx = data["char2idx"]
    idx2char = {int(k): v for k, v in data["idx2char"].items()}
    return char2idx, idx2char


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--out", default="vocab.json")
    args = ap.parse_args()
    build_vocab_from_csv([args.train_csv, args.val_csv, args.test_csv], args.out)
