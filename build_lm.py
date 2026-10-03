"""
build_lm.py

Build a word-level n-gram language model from a Bangla text corpus.

Writes:
    an ARPA file for KenLM / pyctcdecode
    a unigram lexicon (word and count)
    a JSON n-gram table the pure-Python rescoring path can load

KenLM itself is optional. On Windows the kenlm wheel usually does not build;
pyctcdecode then uses the lexicon as unigrams and decode.py keeps the
character-trigram beam search.

Suggested corpora (plain text or a CSV with a text column, not the raw dump):
    Bangla Wikipedia dump:
        https://dumps.wikimedia.org/bnwiki/latest/bnwiki-latest-pages-articles.xml.bz2
    Extract article text first (WikiExtractor or a similar tool), then point
    --corpus at the resulting .txt.
    News text: OSCAR / CC-100 Bangla, or any Bangla news CSV that has a text column.

Usage:
    python build_lm.py --corpus data\\train.csv --order 3 --out data\\word_lm
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter


def read_corpus(path: str) -> list[list[str]]:
    """Return tokenized sentences from a .csv (text column) or a utf-8 text file."""
    sentences: list[list[str]] = []
    if path.lower().endswith(".csv"):
        with open(path, "r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                text = row.get("text") or row.get("sentence") or ""
                words = text.split()
                if words:
                    sentences.append(words)
        return sentences
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            words = line.split()
            if words:
                sentences.append(words)
    return sentences


def count_ngrams(sentences: list[list[str]], order: int) -> list[Counter]:
    """counts[n-1] maps an n-gram tuple to its count. Includes <s> and </s>."""
    counts = [Counter() for _ in range(order)]
    for words in sentences:
        tokens = ["<s>"] * (order - 1) + words + ["</s>"]
        for n in range(1, order + 1):
            for i in range(order - 1, len(tokens)):
                gram = tuple(tokens[i - n + 1 : i + 1])
                if gram[0] == "<s>" and n == 1 and gram != ("<s>",):
                    continue
                counts[n - 1][gram] += 1
        counts[0][("<s>",)] += 1
    return counts


def _log10_prob(count: int, total: int, vocab: int, k: float = 0.1) -> float:
    return math.log10((count + k) / (total + k * max(vocab, 1)))


def write_arpa(counts: list[Counter], path: str) -> None:
    """Write a Kneser-style add-k ARPA file. KenLM can read this when it is installed."""
    order = len(counts)
    vocab = max(len(counts[0]), 1)
    lines = ["\\data\\"]
    for n, bucket in enumerate(counts, start=1):
        lines.append(f"ngram {n}={len(bucket)}")
    lines.append("")
    for n, bucket in enumerate(counts, start=1):
        lines.append(f"\\{n}-grams:")
        context_totals: dict[tuple[str, ...], int] = {}
        if n > 1:
            for gram, count in bucket.items():
                context_totals[gram[:-1]] = context_totals.get(gram[:-1], 0) + count
        unigram_total = sum(counts[0].values())
        for gram, count in sorted(bucket.items(), key=lambda item: (-item[1], item[0])):
            if n == 1:
                prob = _log10_prob(count, unigram_total, vocab)
            else:
                prob = _log10_prob(count, context_totals[gram[:-1]], vocab)
            backoff = "" if n == order else " -0.3"
            lines.append(f"{prob:.6f} {' '.join(gram)}{backoff}")
        lines.append("")
    lines.append("\\end\\")
    lines.append("")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def write_lexicon(counts: list[Counter], path: str, min_count: int = 1) -> int:
    """Write word<TAB>count unigrams, skipping sentence markers."""
    kept = 0
    with open(path, "w", encoding="utf-8") as handle:
        for (word,), count in counts[0].most_common():
            if word in ("<s>", "</s>", "<unk>") or count < min_count:
                continue
            handle.write(f"{word}\t{count}\n")
            kept += 1
    return kept


def write_json(counts: list[Counter], path: str, order: int) -> None:
    payload = {
        "order": order,
        "ngrams": {
            str(n): {" ".join(gram): int(count) for gram, count in bucket.items()}
            for n, bucket in enumerate(counts, start=1)
        },
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a Bangla word n-gram LM")
    ap.add_argument("--corpus", required=True, help="UTF-8 text file, or CSV with a text column")
    ap.add_argument("--order", type=int, default=3, choices=(3, 4, 5))
    ap.add_argument("--out", default="data/word_lm", help="Output path prefix, without extension")
    ap.add_argument("--min_count", type=int, default=1)
    args = ap.parse_args()

    sentences = read_corpus(args.corpus)
    if not sentences:
        raise SystemExit(f"No sentences in {args.corpus}")
    counts = count_ngrams(sentences, args.order)
    arpa_path = args.out + ".arpa"
    lexicon_path = args.out + ".lexicon.txt"
    json_path = args.out + ".json"
    write_arpa(counts, arpa_path)
    n_words = write_lexicon(counts, lexicon_path, min_count=args.min_count)
    write_json(counts, json_path, args.order)
    print(f"Sentences: {len(sentences)}")
    print(f"ARPA order {args.order}: {arpa_path}")
    print(f"Lexicon words: {n_words} -> {lexicon_path}")
    print(f"JSON n-grams: {json_path}")
    print("KenLM binary loading is used automatically when the kenlm package is installed.")


if __name__ == "__main__":
    main()
