"""
decode.py

CTC prefix beam search with a character trigram language model and an
optional word lexicon. The language model is built from training transcripts.

Usage:
    python decode.py --train_csv data\\train.csv --out data\\lm.json
"""

import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np

NEG = -1e30


def logsumexp(a, b):
    if a < b:
        a, b = b, a
    if b < NEG / 2:
        return a
    return a + math.log1p(math.exp(b - a))


def logits_to_log_probs(logits):
    """logits: numpy (T, C) or (B, T, C)."""
    arr = np.asarray(logits, dtype=np.float64)
    m = arr.max(axis=-1, keepdims=True)
    shifted = arr - m
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def _char_of(idx, idx2char):
    if idx in idx2char:
        return idx2char[idx]
    return idx2char.get(str(idx), "")


class CharLM:
    """Smoothed character trigram with bigram and unigram backoff, plus a word lexicon."""

    def __init__(self, contexts, lexicon, vocab_size, alpha=0.2):
        self.contexts = contexts
        self.lexicon = lexicon
        self.vocab_size = max(int(vocab_size), 1)
        self.alpha = alpha
        self._totals = {ctx: sum(bucket.values()) for ctx, bucket in contexts.items()}

    def score(self, prefix, new_idx, idx2char):
        ch = _char_of(new_idx, idx2char)
        if not ch:
            return 0.0
        prev = []
        for idx in prefix[-2:]:
            p = _char_of(idx, idx2char)
            if p:
                prev.append(p)
        return self._logp(prev, ch)

    def _logp(self, prev, ch):
        for n in (2, 1, 0):
            ctx = "".join(prev[-n:]) if n else ""
            bucket = self.contexts.get(ctx)
            if not bucket:
                continue
            total = self._totals[ctx]
            count = bucket.get(ch, 0)
            return math.log((count + self.alpha) / (total + self.alpha * self.vocab_size))
        return -math.log(self.vocab_size)

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload = {
            "contexts": self.contexts,
            "lexicon": self.lexicon,
            "vocab_size": self.vocab_size,
            "alpha": self.alpha,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        lexicon = {word: int(count) for word, count in payload["lexicon"].items()}
        return cls(payload["contexts"], lexicon, payload["vocab_size"], payload.get("alpha", 0.2))


def build_char_lm(csv_path, min_word_count=2):
    import csv

    contexts = defaultdict(lambda: defaultdict(int))
    lexicon = defaultdict(int)
    chars = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = " ".join((row.get("text") or "").split())
            if not text:
                continue
            symbols = list(text)
            chars.update(symbols)
            for i, ch in enumerate(symbols):
                contexts[""][ch] += 1
                if i >= 1:
                    contexts[symbols[i - 1]][ch] += 1
                if i >= 2:
                    contexts[symbols[i - 2] + symbols[i - 1]][ch] += 1
            for word in text.split(" "):
                if word:
                    lexicon[word] += 1

    contexts = {ctx: dict(bucket) for ctx, bucket in contexts.items()}
    lexicon = {word: count for word, count in lexicon.items() if count >= min_word_count}
    return CharLM(contexts, lexicon, vocab_size=max(len(chars), 1))


def ctc_prefix_beam_decode(log_probs, idx2char, beam_width=8, lm=None, lm_weight=0.15, top_k=20):
    """
    log_probs: numpy array (T, C), index 0 is the CTC blank.
    Returns the decoded string.
    """
    arr = np.asarray(log_probs, dtype=np.float64)
    t_steps, n_classes = arr.shape
    beams = {(): [0.0, NEG]}
    k = min(top_k, n_classes)

    for t in range(t_steps):
        row = arr[t]
        choices = np.argpartition(row, -k)[-k:]
        if 0 not in choices:
            choices = np.append(choices, 0)
        updated = {}
        for prefix, (p_blank, p_non) in beams.items():
            p_total = logsumexp(p_blank, p_non)
            for idx in choices:
                idx = int(idx)
                lp = float(row[idx])
                if idx == 0:
                    slot = updated.setdefault(prefix, [NEG, NEG])
                    slot[0] = logsumexp(slot[0], p_total + lp)
                    continue

                bonus = 0.0
                if lm is not None and lm_weight:
                    bonus = lm_weight * lm.score(prefix, idx, idx2char)
                extended = prefix + (idx,)
                slot = updated.setdefault(extended, [NEG, NEG])
                if prefix and prefix[-1] == idx:
                    # Same label again only extends the text after a blank.
                    # A repeat with no blank stays on the current prefix.
                    repeat = updated.setdefault(prefix, [NEG, NEG])
                    repeat[1] = logsumexp(repeat[1], p_non + lp)
                    slot[1] = logsumexp(slot[1], p_blank + lp + bonus)
                else:
                    slot[1] = logsumexp(slot[1], p_total + lp + bonus)

        ranked = sorted(
            ((logsumexp(pb, pn), prefix, pb, pn) for prefix, (pb, pn) in updated.items()),
            reverse=True,
        )
        beams = {prefix: [pb, pn] for _, prefix, pb, pn in ranked[:beam_width]}

    best_prefix = max(beams.items(), key=lambda item: logsumexp(item[1][0], item[1][1]))[0]
    return "".join(_char_of(idx, idx2char) for idx in best_prefix)


def _lexicon_buckets(lexicon):
    buckets = defaultdict(list)
    for word, count in lexicon.items():
        if count < 5 or len(word) < 4:
            continue
        buckets[(len(word), word[0])].append((word, count))
    return buckets


def correct_with_lexicon(text, lm):
    """Replace a word only when exactly one frequent lexicon word is one edit away."""
    if lm is None or not lm.lexicon or not text:
        return text
    buckets = _lexicon_buckets(lm.lexicon)
    pieces = text.split(" ")
    fixed = []
    for word in pieces:
        if not word or word in lm.lexicon or len(word) < 4:
            fixed.append(word)
            continue
        found = []
        for length in (len(word) - 1, len(word), len(word) + 1):
            for candidate, count in buckets.get((length, word[0]), ()):
                if _edit_distance_at_most_one(word, candidate):
                    found.append((count, candidate))
                    if len(found) > 1:
                        break
            if len(found) > 1:
                break
        if len(found) == 1:
            fixed.append(found[0][1])
        else:
            fixed.append(word)
    return " ".join(fixed)


def _edit_distance_at_most_one(a, b):
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    i = j = 0
    edits = 0
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
            continue
        edits += 1
        if edits > 1:
            return False
        if la == lb:
            i += 1
            j += 1
        elif la > lb:
            i += 1
        else:
            j += 1
    edits += (la - i) + (lb - j)
    return edits <= 1


def decode_log_probs(log_probs, idx2char, lm=None, beam_width=8, lm_weight=0.15, lexicon=False):
    text = ctc_prefix_beam_decode(
        log_probs, idx2char, beam_width=beam_width, lm=lm, lm_weight=lm_weight if lm is not None else 0.0
    )
    if lexicon and lm is not None:
        text = correct_with_lexicon(text, lm)
    return text


def default_lm_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "lm.json")


def load_lm(path=None):
    candidates = []
    if path:
        candidates.append(path)
    candidates.append(default_lm_path())
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return CharLM.load(candidate)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--out", default="data/lm.json")
    args = ap.parse_args()
    lm = build_char_lm(args.train_csv)
    lm.save(args.out)
    print(f"Saved character LM with {len(lm.lexicon)} lexicon words to {args.out}")


if __name__ == "__main__":
    main()
