"""Smoke test for the held-out red-ink line sample."""

import os
import sys
import unicodedata
import unittest

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SAMPLE_DIR = os.path.join(ROOT, "tests", "real_samples")
IMAGE = os.path.join(SAMPLE_DIR, "IMG-20261002-WA0000.jpg")
LABEL = os.path.join(SAMPLE_DIR, "IMG-20261002-WA0000.txt")
# U+09DD is the composed form of ড-independent ঢ়.
EXPECTED = unicodedata.normalize("NFC", "কিংকর্তব্যবিমূ\u09dd")


class RealSampleTest(unittest.TestCase):
    def test_label_matches_the_word(self):
        with open(LABEL, "r", encoding="utf-8") as handle:
            self.assertEqual(unicodedata.normalize("NFC", handle.read().strip()), EXPECTED)

    def test_image_loads_as_color(self):
        img = cv2.imread(IMAGE, cv2.IMREAD_COLOR)
        self.assertIsNotNone(img)
        self.assertEqual(img.ndim, 3)
        self.assertGreater(img.shape[0], 8)
        self.assertGreater(img.shape[1], 8)

    def test_harness_pairs_image_with_label(self):
        from evaluate import load_sample_pairs, score_texts

        pairs = load_sample_pairs(SAMPLE_DIR)
        self.assertEqual(len(pairs), 1)
        self.assertTrue(pairs[0][0].endswith("IMG-20261002-WA0000.jpg"))
        self.assertEqual(unicodedata.normalize("NFC", pairs[0][1]), EXPECTED)
        metrics = score_texts(["কিংবর্তব্যবিসুড়"], [EXPECTED])
        self.assertGreater(metrics["cer"], 0.0)
        self.assertEqual(metrics["n"], 1)

    def test_checkpoint_smoke_when_present(self):
        checkpoint = os.path.join(ROOT, "checkpoints", "best.pt")
        if not os.path.isfile(checkpoint):
            self.skipTest("checkpoints/best.pt is not in this checkout")
        import torch

        from evaluate import evaluate_sample_dir
        from recognize import build_crnn_recognizer, load_torch_checkpoint

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = load_torch_checkpoint(checkpoint, device)
        recognizer = build_crnn_recognizer(ckpt, device)
        metrics = evaluate_sample_dir(recognizer.recognize_line_path, SAMPLE_DIR)
        self.assertEqual(metrics["n"], 1)
        self.assertTrue(metrics["samples"][0]["prediction"])
        self.assertGreaterEqual(metrics["cer"], 0.0)
        self.assertLessEqual(metrics["cer"], 1.0)


if __name__ == "__main__":
    unittest.main()
