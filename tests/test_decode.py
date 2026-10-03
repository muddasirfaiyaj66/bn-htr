"""Tests for the character LM, post-correction, and word-LM builder."""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from build_lm import count_ngrams, read_corpus, write_arpa, write_lexicon  # noqa: E402
from decode import (  # noqa: E402
    CharLM,
    correct_with_lm,
    ctc_confidence,
    ctc_prefix_beam_decode,
    describe_decoder,
    pyctc_decode,
    score_text,
)
from recognize import maybe_vlm_correct  # noqa: E402
from normalize import normalize_bangla  # noqa: E402
from vocab import grapheme_clusters  # noqa: E402


class DecodeTest(unittest.TestCase):
    def test_hasanta_stays_inside_one_cluster(self):
        clusters = [c for c in grapheme_clusters("ক্ত") if not c.isspace()]
        self.assertEqual(clusters, ["ক্ত"])
        self.assertEqual([c for c in grapheme_clusters("কি") if not c.isspace()], ["কি"])

    def test_nukta_forms_compose(self):
        self.assertEqual(normalize_bangla("ঢ\u09bc"), "\u09dd")
        self.assertEqual(normalize_bangla("ড\u09bc"), "\u09dc")
        self.assertEqual(normalize_bangla("য\u09bc"), "\u09df")
        self.assertEqual(normalize_bangla("\u09dd"), "\u09dd")

    def test_prefix_beam_reads_a_peaked_path(self):
        log_probs = np.full((5, 3), -8.0)
        log_probs[0, 0] = 0.0
        log_probs[1, 1] = 0.0
        log_probs[2, 1] = 0.0
        log_probs[3, 0] = 0.0
        log_probs[4, 2] = 0.0
        text = ctc_prefix_beam_decode(log_probs, {1: "ক", 2: "খ"}, beam_width=4, lm=None)
        self.assertEqual(text, "কখ")

    def test_post_correct_needs_a_clear_lm_win(self):
        lm = CharLM(
            {
                "": {"অ": 20, "ক": 20, "খ": 1, "গ": 10, "ঘ": 10},
                "অ": {"ক": 30, "খ": 1},
                "ক": {"গ": 10},
                "খ": {"গ": 1},
                "গ": {"ঘ": 10},
            },
            {"অকগঘ": 10},
            vocab_size=5,
        )
        original = "অখগঘ"
        gap = score_text(lm, "অকগঘ") - score_text(lm, original)
        self.assertGreater(gap, 0.0)
        self.assertEqual(correct_with_lm(original, lm, max_dist=2, margin=gap + 1.0), original)
        self.assertEqual(correct_with_lm(original, lm, max_dist=2, margin=gap - 0.01), "অকগঘ")
        self.assertEqual(correct_with_lm(original, lm, max_dist=0, margin=gap - 0.01), original)

    def test_missing_lm_is_described(self):
        text = describe_decoder(None, None)
        self.assertIn("not loaded", text)
        self.assertIn("greedy", text)

    def test_build_lm_writes_arpa_and_lexicon(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = os.path.join(tmp, "corpus.txt")
            with open(corpus, "w", encoding="utf-8") as handle:
                handle.write("কিং কর্তব্য বিমূঢ়\nকিং কর্তব্য\n")
            sentences = read_corpus(corpus)
            counts = count_ngrams(sentences, 3)
            arpa = os.path.join(tmp, "lm.arpa")
            lexicon = os.path.join(tmp, "lm.lexicon.txt")
            write_arpa(counts, arpa)
            n_words = write_lexicon(counts, lexicon)
            self.assertGreaterEqual(n_words, 2)
            with open(arpa, "r", encoding="utf-8") as handle:
                body = handle.read()
            self.assertIn("\\data\\", body)
            self.assertIn("\\end\\", body)
            with open(lexicon, "r", encoding="utf-8") as handle:
                words = handle.read()
            self.assertIn("কিং", words)

    def test_confidence_is_higher_on_a_peaked_path(self):
        peaked = np.full((4, 3), -8.0)
        peaked[:, 1] = -0.05
        flat = np.full((4, 3), -2.0)
        self.assertGreater(ctc_confidence(peaked), ctc_confidence(flat))

    def test_vlm_stays_off_unless_configured(self):
        image = np.full((8, 8), 255, np.uint8)
        with patch.dict(os.environ, {"VLM_API_URL": "", "VLM_API_KEY": ""}, clear=False):
            self.assertIsNone(maybe_vlm_correct(image, "ক"))
        with patch.dict(os.environ, {"VLM_API_URL": "http://127.0.0.1:9/v1", "VLM_API_KEY": ""}, clear=False):
            with patch("urllib.request.urlopen") as opened:
                self.assertIsNone(maybe_vlm_correct(image, "ক"))
                opened.assert_not_called()

    def test_pyctcdecode_accepts_hotwords(self):
        logits = np.full((6, 3), -5.0, dtype=np.float32)
        logits[1, 1] = 6.0
        logits[2, 1] = 6.0
        logits[4, 2] = 6.0
        text = pyctc_decode(logits, {1: "ক", 2: "খ"}, beam_width=8, hotwords=["কখ"], hotword_weight=2.0)
        self.assertIsNotNone(text)
        self.assertIn("ক", text)


if __name__ == "__main__":
    unittest.main()
