"""
vocab.py

Builds and stores the character vocabulary used for CTC training.
Index 0 is reserved for the CTC blank token.
"""

import csv
import json
import unicodedata

from normalize import normalize_bangla

_VIRAMA = "\u09cd"


def grapheme_clusters(text):
    """Bangla orthographic clusters: base, nukta, vowel signs, and hasant groups."""
    text = normalize_bangla(text)
    clusters = []
    i = 0
    while i < len(text):
        if text[i].isspace():
            clusters.append(text[i])
            i += 1
            continue
        j = i + 1
        while j < len(text):
            if text[j] == _VIRAMA and j + 1 < len(text):
                j += 2
                continue
            if unicodedata.category(text[j]) in ("Mn", "Mc"):
                j += 1
                continue
            break
        clusters.append(text[i:j])
        i = j
    return clusters


def grapheme_report(csv_paths):
    """Compare character tokens with grapheme clusters. Does not change the vocab."""
    char_vocab = set()
    cluster_vocab = set()
    char_len = 0
    cluster_len = 0
    conjuncts = 0
    clusters_n = 0
    lines = 0
    for path in csv_paths:
        with open(path, "r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                text = normalize_bangla(row.get("text") or "")
                if not text:
                    continue
                lines += 1
                char_vocab.update(text)
                char_len += len(text)
                clusters = [c for c in grapheme_clusters(text) if not c.isspace()]
                cluster_vocab.update(clusters)
                cluster_len += len(clusters)
                clusters_n += len(clusters)
                conjuncts += sum(1 for c in clusters if _VIRAMA in c)
    return {
        "lines": lines,
        "char_vocab": len(char_vocab),
        "grapheme_vocab": len(cluster_vocab),
        "mean_chars": char_len / max(lines, 1),
        "mean_graphemes": cluster_len / max(lines, 1),
        "conjunct_fraction": conjuncts / max(clusters_n, 1),
        "adopted": False,
        "reason": (
            "A grapheme vocabulary changes every CTC target and needs a full retrain. "
            "Character error rate was not measured, so the character vocabulary stays."
        ),
    }


def build_vocab_from_csv(csv_paths, out_json_path):
    chars = set()
    for path in csv_paths:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                chars.update(normalize_bangla(row["text"]))

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
    ap.add_argument("--graphemes", action="store_true", help="Report grapheme-cluster vocab size and exit")
    args = ap.parse_args()
    paths = [args.train_csv, args.val_csv, args.test_csv]
    if args.graphemes:
        report = grapheme_report(paths)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.out:
            with open(args.out, "w", encoding="utf-8") as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
    else:
        build_vocab_from_csv(paths, args.out)
