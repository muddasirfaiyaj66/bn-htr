"""
decode.py

CTC prefix beam search with a character trigram language model and an
optional word lexicon. The language model is built from training transcripts.

Usage:
    python decode.py --train_csv data\\train.csv --out data\\lm.json
"""

import argparse
import json
import logging
import math
import os
from collections import defaultdict
from functools import lru_cache

import numpy as np

from normalize import normalize_bangla

logger = logging.getLogger(__name__)

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


def find_lm_path(path=None):
    """Return the first existing LM json, or None."""
    candidates = []
    if path:
        candidates.append(path)
    candidates.append(default_lm_path())
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return None


def load_lm(path=None):
    found = find_lm_path(path)
    if not found:
        return None
    return CharLM.load(found)


def describe_decoder(lm, lm_path, beam_width=8, post_correct=False, engine="prefix"):
    """One-line description of the active decoder for logs and the UI."""
    looked_for = lm_path or default_lm_path()
    if lm is None or beam_width <= 1:
        return f"greedy CTC (language model not loaded, looked for {looked_for})"
    name = "pyctcdecode+word-lm" if engine == "pyctc" else "ctc-prefix-beam+char-lm"
    if post_correct:
        name += "+post-correct"
    words = len(getattr(lm, "lexicon", {}) or {})
    return f"{name} ({lm_path}, {words} lexicon words)"


def score_text(lm, text):
    """Sum of character-trigram log probabilities. Higher is more likely."""
    if lm is None or not text:
        return 0.0
    total = 0.0
    prev: list[str] = []
    for ch in normalize_bangla(text):
        total += lm._logp(prev, ch)
        prev.append(ch)
        prev = prev[-2:]
    return total


def _levenshtein_at_most(a, b, limit):
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    try:
        from rapidfuzz.distance import Levenshtein

        return int(Levenshtein.distance(a, b, score_cutoff=limit))
    except ImportError:
        if limit <= 1:
            return 0 if _edit_distance_at_most_one(a, b) else 2
        # Small bounded distance without rapidfuzz.
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, start=1):
            cur = [i]
            row_min = i
            for j, cb in enumerate(b, start=1):
                ins = cur[j - 1] + 1
                delete = prev[j] + 1
                sub = prev[j - 1] + (ca != cb)
                best = ins if ins < delete else delete
                if sub < best:
                    best = sub
                cur.append(best)
                if best < row_min:
                    row_min = best
            if row_min > limit:
                return limit + 1
            prev = cur
        return prev[-1]


def correct_with_lm(text, lm, max_dist=2, margin=1.0, min_count=5):
    """
    Replace an out-of-lexicon word only when a nearby lexicon word's LM score
    beats the original by `margin`. Bangla nukta variants are compared equal.
    """
    if lm is None or not getattr(lm, "lexicon", None) or not text:
        return text
    pieces = text.split(" ")
    fixed = []
    lexicon = lm.lexicon
    for word in pieces:
        key = normalize_bangla(word)
        if not key or key in lexicon or len(key) < 4:
            fixed.append(word)
            continue
        best_word = None
        best_score = None
        original_score = score_text(lm, key)
        for candidate, count in lexicon.items():
            if int(count) < min_count:
                continue
            norm = normalize_bangla(candidate)
            if not norm or abs(len(norm) - len(key)) > max_dist:
                continue
            if _levenshtein_at_most(key, norm, max_dist) > max_dist:
                continue
            cand_score = score_text(lm, norm)
            if best_score is None or cand_score > best_score:
                best_score = cand_score
                best_word = candidate
        if best_word is not None and best_score is not None and best_score > original_score + margin:
            fixed.append(best_word)
        else:
            fixed.append(word)
    return " ".join(fixed)


def _label_list(idx2char):
    indexes = [int(i) for i in idx2char]
    size = max(indexes) + 1 if indexes else 1
    labels = [""] * size
    labels[0] = ""
    for idx, ch in idx2char.items():
        labels[int(idx)] = ch
    return labels


@lru_cache(maxsize=8)
def _cached_pyctc(labels, kenlm_path, unigrams, alpha, beta):
    from pyctcdecode import build_ctcdecoder

    return build_ctcdecoder(
        list(labels),
        kenlm_path or None,
        unigrams=list(unigrams) if unigrams else None,
        alpha=alpha,
        beta=beta,
    )


def pyctc_decode(logits, idx2char, kenlm_path=None, unigrams=None, alpha=0.5, beta=1.5, beam_width=16, hotwords=None, hotword_weight=10.0):
    """
    Decode raw CTC logits with pyctcdecode.

    Returns None when pyctcdecode is not installed so the caller can fall
    back to the prefix beam search. KenLM is optional; unigrams still apply.
    """
    try:
        import pyctcdecode  # noqa: F401
    except ImportError:
        logger.info("pyctcdecode is not installed. Falling back to the prefix beam search.")
        return None
    labels = tuple(_label_list(idx2char))
    grams = tuple(unigrams or ())
    try:
        decoder = _cached_pyctc(labels, kenlm_path or "", grams, float(alpha), float(beta))
        return decoder.decode(
            np.asarray(logits, dtype=np.float32),
            beam_width=int(beam_width),
            hotwords=list(hotwords) if hotwords else None,
            hotword_weight=float(hotword_weight),
        )
    except Exception as exc:  # missing kenlm bindings or an unsupported LM file
        logger.info("pyctcdecode could not run (%s). Falling back to the prefix beam search.", exc)
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
